import os

import requests
import streamlit as st
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from flask import Flask

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

assert endpoint, "Please set the AZURE_OPENAI_ENDPOINT environment variable"
assert deployment_name, "Please set the DEPLOYMENT_NAME environment variable"
assert api_version, "Please set the AZURE_OPENAI_API_VERSION environment variable"

def generate_structured_response(sys_msgs, hmn_msg):
    llm = AzureChatOpenAI(
        api_version=api_version,  # type: ignore
        azure_deployment=deployment_name,
    )
    chain = None
    for sys_msg in sys_msgs:
        prompt = ChatPromptTemplate.from_messages(
            [("system", sys_msg), ("human", hmn_msg)]
        )
        chain = prompt | llm
        hmn_msg = chain.invoke({}).content
    return hmn_msg

def chain_prompts(text_input):
    MAIN_SUBJECTS_SYS_MESSAGE = "אתה מפשט לשוני מקצועי שעוזר לאנשים עם הנמכה קוגניטיבית להבין טקסטים מורכבים בצורה פשוטה. לאנשים עם מגבלה קוגניטיבית מתקשים להבין טקסטים מורכבים וצריכים הפשטה. כדי לעשות זאת, עליך להוציא את הרעיונות העיקריים של הטקסט. הוצא את הרעיונות המרכזיים של הטקסט הבא:"
    SIMPLIFY_LANGUAGE_SYS_MESSAGE = "אתה מפשט לשוני מקצועי שעוזר לאנשים עם הנמכה קוגניטיבית להבין טקסטים מורכבים בצורה פשוטה. לאנשים עם מגבלה קוגניטיבית מתקשים להבין טקסטים מורכבים וצריכים הפשטה. כדי לעשות זאת, תקבל רעיונות מרכזיים מטקסט, תנסח כל משפט בצורה פשוטה לפי החוקים הבאים:"
    "סדר את המשפט בסדר הגיוני, כך שהמידע החשוב בתחילת המשפט ולא בסופו והמשפט בסדר הגיוני."
    "דוגמה: מקור: 'תעלה לרכבת אחרי שתקנה כרטיס בקופה'. פישוט: 'תקנה כרטיס בקופה ואחר כך תעלה לרכבת'"
    "שמור על עקביות: בכל פעם שמתייחסים למונח, השתמש באותה מילה והקפד על שימוש עקבי בנטיית המין, המספר, הגוף והזמנים הדקדוקיים."
    "השתמש בפעלים במקום בשמות פעולה."
    "דוגמה: מקור: שתייה חשובה לשמירה על הבריאות. פישוט: חשוב לשתות כדי לשמור על הבריאות."
    "השתמש בצורת הפועל פעיל ולא סביל."
    "דוגמה: מקור: המכונית נמכרה על-ידי הסוחר. פישוט: הסוחר מכר את המכונית."
    "דוגמה: מקור: בפנותך על המשרד, פישוט: כאשר אתה פונה אל המשרד. מקור: מתי מגיע תורך? פישוט: מתי מגיע התור שלך?"
    "השתמש במשפטים קצרים ופשוטים מבחינה תחבירית והפרד משפטים מורכבים למספר משפטים פשוטים תוך שמירה על משמעות המשפט המקורי."
    "דוגמה: מקור: הפקיד שאיתו דיברתי אתמול עובד בביטוח לאומי, פישוט: אתמול דיברתי עם הפקיד. הפקיד הזה עובד בביטוח לאומי."
    "השתמש בסדר הפועלים: נושא - פועל - מושא."
    "דוגמה: מקור: את הסיבה לאיחור גילו המדריכים. פישוט: המדריכים גילו את הסיבה לאיחור."
    SIMPLIFY_WORDS_SYS_MESSAGE = "אתה מפשט לשוני מקצועי שעוזר לאנשים עם הנמכה קוגניטיבית להבין טקסטים מורכבים בצורה פשוטה. לאנשים עם מגבלה קוגניטיבית מתקשים להבין טקסטים מורכבים וצריכים הפשטה. כדי לעשות זאת, תקבל מספר משפטים, כאשר יש מילה מורכבת, מילה שאינה נפוצה או לא ברורה, תחליף אותה במילה פשוטה. אם אין לה חלופה פשוטהף תסביר אותה במילים פשוטות מיד לאחר הופעת המילה :"
    "השתדל להימנע משימוש במושגים מופשטים. אם בכל זאת תשתמש במושג מופשט, לווה אותו בהסבר או בדוגמה מתאימה."
    "דוגמה: מקור: אסור להפלות אנשים עם מוגבלות. פישוט: אסור להפלות אנשים עם מוגבלות. זאת אומרת, אסור להתנהג אל אנשים עם מוגבלות בצורה שונה או פחות טובה."
    "דוגמה: מקור: מה לעשות בעת קבלת התרעה על ירי רקטות וטילים, פישוט: חשוב שנדע מה עושים כשיורים עלינו רקטות וטילים. איך אנחנו יודעים שיש ירי של רקטות וטילים? אנחנו מקבלים התרעה. התרעה היא אזהרה שקורה עכשיו משהו מסוכן. את ההתרעה אנחנו מקבלים בצורות שונות: לפעמים נשמע אזעקה (קול חזק מרמקולים שנמצאים בחוץ). לפעמים תהיה הודעה בטלוויזיה או ברדיו שיש ירי של רקטות וטילים. כדאי מאוד להוריד את אפליקציית פיקוד העורף ואז נוכל גם שם לקבל התרעה."
    "השתמש באוצר מילים יומיומי, שכיח ושגור."
    "דוגמה: מקור: ממתין, פישוט: מחכה. מקור: מזהה, פישוט: מכיר. מקור: מסתיים, פישוט: נגמר. מקור: יתכן שיתרחשו, פישוט: אולי יהיו עוד. מקור: האזינו, פישוט: תקשיבו."
    "הימנעו משימוש בראשי תיבות, בקיצורים ובסלנג. אם יש צורך להשתמש בהם, הסבירו אותם."
    "דוגמה: מקור: ממ'ד, פישוט: מרחב מוגן דירתי."
    "הפרד כינויי שייכות. דוגמה: מקור: שמו, פישוט: השם שלו. מקור: בריאותך, פישוט: הבריאות שלך"
    "השתמש בצורת הפעיל או בהנחייה במקום ציווי. מקור: שתו, פישוט: צריך לשתות. מקור: הישארו בחוץ. פישוט: צריך להישאר בחוץ"
    FINAL_VALIDATION_MESSAGE = "אתה מפשט לשוני מקצועי שעוזר לאנשים עם הנמכה קוגניטיבית להבין טקסטים מורכבים בצורה פשוטה. לאנשים עם מגבלה קוגניטיבית מתקשים להבין טקסטים מורכבים וצריכים הפשטה"
    "אתה תקבל טקסט שעבר פישוט, ותתקן אותו כדי שיהיה ברור לקוראים ויעמוד בהנחיות:"
    "המשפטים קצרים, בנויים ברצף לוגי ברור, וניתנים לקריאה רציפה. אם המשפט ארוך פצל אותו לשני משפטים"
    "שימוש באוצר מילים שכיח ויום-יומי. אם יש מילה לא ברורה, החלף אותה במילה ברורה."
    sys_msgs = [
        MAIN_SUBJECTS_SYS_MESSAGE,
        SIMPLIFY_LANGUAGE_SYS_MESSAGE,
        SIMPLIFY_WORDS_SYS_MESSAGE,
        FINAL_VALIDATION_MESSAGE,
    ]
    result = generate_structured_response(sys_msgs, text_input)
    return result

app = Flask(__name__)

@app.route("/", methods=["POST"])
def handle_chain_prompts():
    data = request.json
    text_input = data.get("text_input", "")
    result = chain_prompts(text_input)
    return jsonify({"result": result})

# Streamlit app
st.set_page_config(page_title="Text Simplification App", layout="centered")
st.markdown(
    """
    <style>
    .reportview-container {
        text-align: right;
        font-size: 20px;
        background-color: #f0f0f5;
        color: #333;
    }
    .stTextInput, .stTextArea {
        text-align: right;
        font-size: 18px;
    }
    .stButton button {
        font-size: 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("פישוט לשוני")

api_key = st.text_input("Enter your Azure OpenAI API Key", type="password")
text_input = st.text_area("הכניסו כאן את הטקסט המסובך")

if st.button("Simplify Text"):
    if api_key and text_input:
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"text_input": text_input}
        response = requests.post(
            "https://pashutlinux.azurewebsites.net/chain_prompts", headers=headers, json=data
        )
        if response.status_code == 200:
            st.write("Simplified Text:")
            st.write(response.json().get("result", "No response from backend"))
        else:
            st.error("Error from backend")
    else:
        st.error("Please provide both API key and text input")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)