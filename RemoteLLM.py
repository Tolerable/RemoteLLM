import os
from transformers import GPTNeoXForCausalLM, AutoTokenizer, TextStreamer, StoppingCriteria, StoppingCriteriaList
import torch
import transformers

model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-1.4b",
    revision="step143000",
    cache_dir="./pythia-1.4b/step143000",
)

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-1.4b",
    revision="step143000",
    cache_dir="./pythia-1.4b/step143000",
)
stop_token_ids = tokenizer.convert_tokens_to_ids([""])
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])
streamer = TextStreamer(tokenizer)
while True:
    user_input = input("Your input: ")
    print(f"Query: {user_input} Response:\n")  # Print the user's query with a line break before the response

    prompt = 'Query: ' + user_input + ' Response: '
    input_ids = tokenizer(prompt, return_tensors="pt")
    
    # Define attention_mask
    attention_mask = torch.ones(input_ids["input_ids"].shape, dtype=torch.long)

    generation_output = model.generate(
        input_ids=input_ids["input_ids"],
        attention_mask=attention_mask,  # Use the defined attention mask
        do_sample=True,
        max_new_tokens=500,  # Reduced max new tokens to control output length
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        repetition_penalty=1.25,
        top_p=0.95,  # Adjusted to control randomness
        temperature=0.7,  # Adjusted to control randomness
        top_k=50,  # Adjusted to control randomness
        stopping_criteria=stopping_criteria,
    )

    # Convert the generated output to a list of integers
    generated_output_ids = generation_output.tolist()

    # Decode the list of integers into a string
    generated_text = tokenizer.decode(generated_output_ids[0], skip_special_tokens=True)

    # Print the generated text with a line break for readability
    print(generated_text + "\n")
