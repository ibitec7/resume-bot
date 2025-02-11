import ollama

user_input = "What skills does the candidate have on his resume?"
user_input = "What is the job description?"
context = ollama.chat(model = "qwen2.5:3b",messages=[
                        {"role": "system", "content": f"Tell me if the following question requires\
                             the context of the job description, the resume, or both.\
                                 Answer by saying one word that is 'job', 'resume', or 'both' in lowercase.\
                                     Question: {user_input}"},
                    ])


print(context.message.content)
