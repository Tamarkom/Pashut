import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

load_dotenv()

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

assert endpoint, "Please set the AZURE_OPENAI_ENDPOINT environment variable"
assert deployment_name, "Please set the DEPLOYMENT_NAME environment variable"
assert api_version, "Please set the AZURE_OPENAI_API_VERSION environment variable"

def generate_structured_response(sys_msgs, hmn_msg, api_key):
    print("System messages:", sys_msgs)  # Debug log
    print("Human message:", hmn_msg)  # Debug log
    llm = AzureChatOpenAI(
        api_version=api_version,  # type: ignore
        azure_deployment=deployment_name,
        azure_api_key=api_key,  # Use the API key from the request
    )
    chain = None
    for sys_msg in sys_msgs:
        prompt = ChatPromptTemplate.from_messages(
            [("system", sys_msg), ("human", hmn_msg)]
        )
        chain = prompt | llm
        hmn_msg = chain.invoke({}).content
    return hmn_msg

def chain_prompts(text_input, api_key):
    print("Received text_input:", text_input)  # Debug log
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
    result = generate_structured_response(sys_msgs, text_input, api_key)
    print("Generated result:", result)  # Debug log
    return result

app = Flask(__name__)

@app.route('/api/endpoint', methods=['POST'])
def handle_chain_prompts():
    print("Request received:", request.json)  # Debug log
    data = request.json
    print("Received data:", data)  # Debug log to confirm the payload
    text_input = data.get("text_input", "")
    api_key = data.get("api_key", "")  # Get the API key from the request payload

    if not api_key:
        print("Error: API key is missing")  # Debug log for missing API key
        return jsonify({"error": "API key is missing"}), 400

    result = chain_prompts(text_input, api_key)
    print("Response to send:", result)  # Debug log
    return jsonify({"result": result})

@app.route('/')
def serve_frontend():
    # Serve the main page of the Streamlit app
    return send_from_directory('frontend', 'index.html')


if __name__ == '__main__':
    threading.Thread(target=run_streamlit).start()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
