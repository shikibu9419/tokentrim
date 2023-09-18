import tiktoken
from typing import List, Dict, Any, Tuple, Optional, Union
from copy import deepcopy

MODEL_MAX_TOKENS = {
  'gpt-4': 8192,
  'gpt-4-0613': 8192,
  'gpt-4-32k': 32768,
  'gpt-4-32k-0613': 32768,
  'gpt-3.5-turbo': 4096,
  'gpt-3.5-turbo-16k': 16384,
  'gpt-3.5-turbo-0613': 4096,
  'gpt-3.5-turbo-16k-0613': 16384,
  'code-llama': 1048, # I'm not sure this is correct.
}

NUM_TOKENS_OFFSET = 3

def get_encoding(model: Optional[str]):
  # Attempt to get the encoding for the specified model
  try:
    return tiktoken.encoding_for_model(model or "cli100k_base")
  except KeyError:
    return tiktoken.get_encoding("cl100k_base")


def get_tokens_per_name_and_message(model: Optional[str]) -> Tuple[int, int]:
  if model in [
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-4-0613",
    "gpt-4-32k-0613",
  ]:
    return (1, 3)
  elif model == "gpt-3.5-turbo-0301":
    return (-1, 4)
  else:
    # Slightly raised numbers for an unknown model / prompt template
    # In the future this should be customizable
    return (2, 4)


def num_tokens_from_messages(messages: List[Dict[str, Any]],
                             model) -> int:
  """
  Function to return the number of tokens used by a list of messages.
  """

  if model == "gpt-3.5-turbo":
    model = "gpt-3.5-turbo-0613"
  if model == "gpt-4":
    model = "gpt-4-0613"

  encoding = get_encoding(model)
  (tokens_per_name, tokens_per_message) = get_tokens_per_name_and_message(model)

  # Calculate the number of tokens
  num_tokens = 0
  for message in messages:
    num_tokens += tokens_per_message
    for key, value in message.items():
      try:
        num_tokens += len(encoding.encode(str(value)))
        if key == "name":
          num_tokens += tokens_per_name
      except:
        print(f"Failed to parse '{key}'.")
        pass

  num_tokens += NUM_TOKENS_OFFSET

  return num_tokens


def get_trimmed_message(_message: Dict[str, Any], tokens_needed: int,
                                 model) -> Dict[str, Any]:
  """
  Shorten a message to fit within a token limit by removing characters from the middle.
  """
  # Make a deep copy of the message to avoid modifying the original
  message = deepcopy(_message)

  # If the limit is exceeded only by the token offset of the message, return without trimming
  (_, tokens_per_message) = get_tokens_per_name_and_message(model)
  if tokens_per_message + NUM_TOKENS_OFFSET > tokens_needed:
    return message

  encoding = get_encoding(model)

  while True:
    content = message["content"]

    total_tokens = num_tokens_from_messages([message], model)
    if total_tokens <= tokens_needed:
      break

    tokens = encoding.encode(content)
    new_length = len(tokens) * (tokens_needed) // total_tokens

    half_length = new_length // 2
    (left_half, right_half) = (tokens[:half_length], tokens[-half_length:])
    trimmed_content = encoding.decode(left_half) + '...' + encoding.decode(right_half)

    message["content"] = trimmed_content

  return message

def trim(
  messages: List[Dict[str, Any]],
  model = None,
  system_message: Optional[str] = None,
  trim_ratio: float = 0.75,
  return_response_tokens: bool = False,
  max_tokens = None
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
  """
    Trim a list of messages to fit within a model's token limit.

    Args:
        messages: Input messages to be trimmed. Each message is a dictionary with 'role' and 'content'.
        model: The OpenAI model being used (determines the token limit).
        system_message: Optional system message to preserve at the start of the conversation.
        trim_ratio: Target ratio of tokens to use after trimming. Default is 0.75, meaning it will trim messages so they use about 75% of the model's token limit.
        return_response_tokens: If True, also return the number of tokens left available for the response after trimming.
        max_tokens: Instead of specifying a model or trim_ratio, you can specify this directly.

    Returns:
        Trimmed messages and optionally the number of tokens available for response.
    """

  # Initialize max_tokens
  if max_tokens == None:

    # Check if model is valid
    if model not in MODEL_MAX_TOKENS:
      raise ValueError(f"Invalid model: {model}. Specify max_tokens instead")

    max_tokens = int(MODEL_MAX_TOKENS[model] * trim_ratio)

  # Deduct the system message tokens from the max_tokens if system message exists
  if system_message:

    system_message_event = {"role": "system", "content": system_message}
    system_message_tokens = num_tokens_from_messages([system_message_event],
                                                     model)

    if system_message_tokens > max_tokens:
      print("`tokentrim`: Warning, system message exceeds token limit, which is probably undesired. Trimming...")

      system_message_event = get_trimmed_message(system_message_event, max_tokens, model)
      system_message_tokens = num_tokens_from_messages([system_message_event],
                                                     model)

    max_tokens -= system_message_tokens

  if max_tokens < 0:
    raise ValueError("TOKEN TRIMMING FAILED: Token limit is exceeded for only trimmed system message and function call messages.")

  final_messages = []
  tokens_remaining = max_tokens

  # Process the messages from latest
  for message in messages[::-1]:
    raw_message_tokens = num_tokens_from_messages([message], model)

    if raw_message_tokens > tokens_remaining:
      if "function_call" not in message:
        # If adding the next message exceeds the token limit, try trimming it
        # (This only works for non-function call messages)
        message = get_trimmed_message(message, tokens_remaining, model)

        # If trimming the message still exceeds the token limit, ignore the rest of the messages
        if num_tokens_from_messages([message], model) > tokens_remaining:
          break

    # Add the message to the start of final_messages
    final_messages.insert(0, message)
    tokens_remaining -= raw_message_tokens


  # Add system message to the start of final_messages if it exists
  if system_message:
    final_messages = [system_message_event] + final_messages

  if return_response_tokens:
    return final_messages, tokens_remaining
  else:
    return final_messages

