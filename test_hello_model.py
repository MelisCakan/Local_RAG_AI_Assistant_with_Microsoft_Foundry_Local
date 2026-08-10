from foundry_local_sdk import Configuration, FoundryLocalManager

#Start Foundry Local
config = Configuration(app_name="hello_model")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

#Choose a small model for testing
model = manager.catalog.get_model("qwen2.5-0.5b")

#Download the model
print("Model downloading...")
model.download(
    lambda p: print(f"\rDownloading: {p:.0f}%", end="", flush=True)
)
print()

#Load the model
print("Loading model...")
model.load()

#Create a chat client
client = model.get_chat_client()

#Send a request to the model and stream the response
print("\nModel response:")
for chunk in client.complete_streaming_chat([
    {
        "role": "user",
        "content": "Hello, world! Please complete this greeting."
    }
]):
    #Check if the chunk has choices and print the content if it does
    if chunk.choices:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)

print()

#Unload the model
model.unload()

