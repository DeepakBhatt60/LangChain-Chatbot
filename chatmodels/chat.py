from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(
    model="ministral-3b-2512",
    temperature=0.7
)

messages = [
     SystemMessage(content="Acts as my girlfriend")]

print("---------------- welcome type 0 to exit the application ----------------")

while True:
    prompt = input("You : ")

    if prompt == "0":
        break

    messages.append(prompt)

    response = model.invoke(messages)

    print("Bot :", response.content)

    messages.append(response.content)

print("\nChat History:")
print(messages)