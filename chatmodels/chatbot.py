from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 64,
        "do_sample": False,
        "repetition_penalty": 1.03,
    },
)

model = ChatHuggingFace(llm=llm)

messages = []

print("---------------- welcome type 0 to exit the application ----------------")

while True:

    prompt = input("You : ")

    if prompt == "0":
        break

    # User message ko history mein add karo
    messages.append(HumanMessage(content=prompt))

    # Puri conversation model ko bhejo
    response = model.invoke(messages)

    # AI ka response print karo
    print("Bot :", response.content)

    # AI response ko bhi history mein save karo
    messages.append(AIMessage(content=response.content))

print("\nChat History:")
for message in messages:
    print(message)