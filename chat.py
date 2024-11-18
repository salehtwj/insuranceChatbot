from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import streamlit as st
import requests

def getweather(city):
    response = requests.get(f"http://api.weatherstack.com/current?access_key=87aad240097b96664b7a2fe4501e4162&query={city}")

    if response.status_code == 200:
        data = response.json()
        if "current" in data:
            data['current']['weather_icons'] = ''
            curr = data['current']
            return curr
        else:
            return "Weather data not found!"
    else:
        return "Error: Unable to fetch data."

llm = ChatGroq(temperature=0.8, groq_api_key="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", model_name="llama3-70b-8192")

PROMPT_TEMPLATE = """
<|SYSTEM|>
You are an AI model working as an employee at Bupa, specializing in predicting climate-related health risks such as heatwaves, storms, and air pollution, and their potential impact on public health. Your primary goal is to help clients understand how changing weather patterns could affect their health and provide personalized health recommendations. Additionally, you will promote Bupa’s insurance coverage to ensure that clients are protected from these risks.

Always highlight:
- The impact of climate conditions on health,
- Risk mitigation strategies, such as preventive measures and medical advice,
- Bupa's insurance services and how they can help clients manage climate-related health risks,
- Tailored recommendations for individuals based on the specific climate risks in their region.

Begin by thanking the client for trusting you to evaluate their health-related climate risks and informing them about the protective measures offered by Bupa’s insurance. Avoid repeating the user's input verbatim; instead, focus on analyzing the weather data and offering actionable health advice, while subtly promoting Bupa’s relevant insurance products.

Make sure to keep responses concise, informative, and actionable.
Answer in the same language the user used!
<|END_SYSTEM|>

<|CONTEXT|>
Current weather in the user’s city: {weather}

Conversation history for context:
{history}
<|END_CONTEXT|>

<|USER|>
{input}
<|END_USER|>

<|ASSISTANT|> 
"""

prompt_template = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["weather", "history", "input"]
)

MODEL = LLMChain(llm=llm,
                 prompt=prompt_template,
                 verbose=False)

history = []

def chat(userInput):
    res = MODEL.invoke({"weather": getweather('riyadh'), "input": userInput, "history": history})
    history.append({
        "user": userInput,
        "assistant": res['text']
    })
    return res['text']

def main():
    st.title("Bupa Climate-Related Health Risk Chatbot")

    st.write("Welcome to the Climate Health Risk Predictor! I can help you understand how climate conditions might impact your health and provide personalized recommendations.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Enter your message:")

    if user_input:
        ai_response = chat(user_input)
        
        st.session_state.messages.append(("user", user_input))
        st.session_state.messages.append(("assistant", ai_response))

    for message in st.session_state.messages:
        if message[0] == "user":
            st.markdown(f"<p style='color:blue; text-align:left'>{message[1]}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='color:green; text-align:right'>{message[1]}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
