"""Lightweight LLM service with async support."""

import asyncio # Why asyncio? Because we are using asyncio for the asynchronous operations.
import logging
from typing import AsyncGenerator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
# Why CausalLM? Because we are using a causal language model for text generation.
# Causality comes from the fact that the model is trained to predict the next token based on the previous tokens.
# Why TextIteratorStreamer? Because we are using a streaming response for the LLM.
from threading import Thread

from app.config import settings
from app.exceptions import ModelLoadError

logger = logging.getLogger(__name__)


class LLMService:
    """Lightweight LLM service using modern async patterns."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._is_loaded = False
    
    async def load_model(self) -> None:
        """Load the lightweight LLM model asynchronously."""
        try:
            logger.info(f"Loading LLM model: {settings.llm_model_name}")
            logger.info(f"Device: {settings.device} (GPU: {settings.use_gpu})")
            
            # Log GPU status
            settings.log_gpu_status()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.llm_model_name,
                use_fast=True, # Fast tokenization method
                trust_remote_code=False
            )
            
            # Add padding token if it doesn't exist, padding token is used to pad the input tokens to the same length.
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine optimal torch dtype and device map
            if settings.use_gpu and torch.cuda.is_available():
                torch_dtype = torch.float16  # Use half precision for GPU, half precision is used to reduce the memory usage.
                device_map = "auto"  # Let transformers handle device placement
                logger.info("Using GPU with half precision (float16)")
            else:
                torch_dtype = torch.float32  # Use full precision for CPU, full precision is used to get the best performance. why not half precision? because half precision is not supported on CPU.
                device_map = None
                logger.info("Using CPU with full precision (float32)")
            
            # Load model with optimal settings
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.llm_model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                use_cache=True  # Enable KV cache for better performance
            )
            
            # Ensure model is on the correct device
            if not settings.use_gpu or device_map is None:
                self.model = self.model.to(settings.device) # Move the model to the correct device. 
            
            # Log memory usage
            if settings.use_gpu and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3 # why 1024**3? because 1024 is the number of bytes in a kilobyte, and 3 is the number of kilobytes in a megabyte.
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
            
            self._is_loaded = True
            logger.info("LLM model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM model: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")
    
    async def generate_response(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Generate a response for the given prompt.
        max_tokens is the maximum number of tokens to generate.
        temperature is the temperature of the model.
        """
        if not self.is_loaded: 
            await self.load_model() # await is used to wait for the model to be loaded. There is a possibility that the model is not loaded yet.
        
        max_tokens = max_tokens or settings.max_tokens
        temperature = temperature or settings.temperature
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", # pt = pytorch tensor
                truncation=True, # truncation is used to truncate the input tokens to the same length. Truncation is the act of shortening something
                max_length=512 # max_length is the maximum length of the input tokens
            ).to(settings.device) # to is used to move the input tokens to the correct device.
            
            # Generate response
            with torch.no_grad(): # no_grad is used to disable gradient calculation. As we use the model for inference.
                outputs = self.model.generate(
                    **inputs, # ** is used to unpack the input tokens. why? because the model.generate function expects a dictionary of input tokens (keys are input_ids, attention_mask, etc.). and it is not a dictionary but a tensor.
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True, # do_sample is used to sample the next token from the model. As opposed to greedy sampling. Greedy sampling is the act of selecting the most likely next token. (maximum likelihood estimation)
                    pad_token_id=self.tokenizer.eos_token_id, #TODO: why not use pad_token_id?
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1 # repetition_penalty is used to penalize the model for repeating the same token. It is used to reduce the probability of the repeated token.
                )
            
            # Decode response
            response = self.tokenizer.decode( # decode is used to decode the output tokens to a string. Reverse mapping of the tokens to the original text.
                outputs[0][inputs['input_ids'].shape[1]:],  # [inputs['input_ids'].shape[1]:] is used to get the output tokens after the input tokens.
                skip_special_tokens=True # skip_special_tokens is used to skip the special tokens. As we don't want to include the special tokens in the response such as <s>, </s>, etc.
            )
            
            return response.strip() # strip is used to remove the leading and trailing whitespace from the response.
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise Exception(f"Failed to generate response: {e}")  
    
    async def generate_stream(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response for the given prompt.
         This method returns AsyncGenerator[str, None] which is a generator that yields strings asynchronously."""
        if not self.is_loaded:
            await self.load_model() # await is used to wait for the model to be loaded. There is a possibility that the model is not loaded yet.
        
        max_tokens = max_tokens or settings.max_tokens
        temperature = temperature or settings.temperature
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(settings.device)
            
            # Create streamer
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                skip_prompt=True, 
                skip_special_tokens=True
            )
            
            # Generation parameters - unpack inputs properly
            generation_kwargs = {
                **inputs,  # Unpack input_ids, attention_mask, etc.
                "streamer": streamer,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "repetition_penalty": 1.1
            }
            
            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield tokens as they're generated
            for token in streamer:
                yield token # yield is used to yield the token as it is generated.
            
            thread.join() # join is used to wait for the thread to finish.
            
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error: {str(e)}"
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    async def unload_model(self) -> None:
        """Unload the model to free memory.
        When we are done with the model, we can unload it to free the memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_loaded = False
        
        # Clear CUDA cache if using GPU
        if settings.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded successfully")
