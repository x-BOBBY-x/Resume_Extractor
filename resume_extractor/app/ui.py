import streamlit as st
import os
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from extractor import extract_resume_data_ollama

st.title("Resume Skill Extractor")

st.markdown("**Extraction method:** LLM (Ollama)")
ollama_model = st.text_input("Ollama Model (default: mistral)", value="mistral", help="Model must be pulled in Ollama. Example: mistral, llama2, phi, etc.")

os.makedirs("data/resumes", exist_ok=True)

try:
    conn = mysql.connector.connect(
        host=os.environ.get("DB_HOST", "localhost"),
        user=os.environ.get("DB_USER", "root"),
        password=os.environ.get("DB_PASSWORD", "password"),
        database=os.environ.get("DB_NAME", "resume_db")
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS resumes (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255),
            email VARCHAR(255),
            phone VARCHAR(50),
            skills TEXT,
            experience TEXT
        )
    """)
except mysql.connector.Error as err:
    st.error(f"Database connection failed: {err}")
    st.stop()

if st.button("Clear All Data"):
    cursor.execute("DELETE FROM resumes")
    conn.commit()

    for f in os.listdir("data/resumes"):
        try:
            os.remove(os.path.join("data/resumes", f))
        except Exception:
            pass
    st.success("All resume data and uploaded files have been cleared.")

uploaded_file = st.file_uploader("Upload a PDF Resume", type="pdf")

if uploaded_file:
    file_path = os.path.join("data/resumes", uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
    except Exception as e:
        st.error(f"Failed to save uploaded file: {e}")
        st.stop()

    try:
        data = extract_resume_data_ollama(file_path, ollama_model)
    except Exception as e:
        st.error(f"Failed to extract resume data: {e}")
        st.stop()

    name = data.get("name", "")
    email = data.get("email", "")
    phone = data.get("phone", "")
    experience = data.get("experience", "")
    skills = data.get("skills", [])
    
    if isinstance(skills, str):
        import ast
        try:
            skills_list = ast.literal_eval(skills)
            if isinstance(skills_list, list):
                skills = skills_list
            else:
                skills = [skills]
        except Exception:
            skills = [s.strip() for s in skills.split(",") if s.strip()]
    if not isinstance(skills, list):
        skills = [str(skills)]

    if not skills:
        st.warning("No skills extracted. Raw LLM output: {}".format(data))

    skills_str = ", ".join(skills)

    cursor.execute("SELECT COUNT(*) FROM resumes WHERE email = %s", (email,))
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
            INSERT INTO resumes (name, email, phone, skills, experience)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, email, phone, skills_str, experience))
        conn.commit()
        st.success("Resume uploaded and data saved to MySQL database!")
    else:
        st.warning("A resume with this email already exists in the database.")

st.subheader("Stored Resumes")
cursor.execute("SELECT name, email, phone, skills, experience FROM resumes")
rows = cursor.fetchall()
df = pd.DataFrame(rows, columns=["Name", "Email", "Phone", "Skills", "Experience"])

if not df.empty:
    all_skills = sorted(set(skill.strip() for skills in df["Skills"] for skill in skills.split(", ")))
    selected_skill = st.selectbox("Filter by Skill", options=["All"] + all_skills)

    if selected_skill != "All":
        df = df[df["Skills"].str.contains(selected_skill)]

    st.dataframe(df)

    chart_data = df["Skills"].str.split(", ").explode().value_counts()
    fig, ax = plt.subplots()
    chart_data.plot(kind="bar", ax=ax)
    ax.set_title("Skill Frequency Across Resumes")
    ax.set_xlabel("Skill")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "resumes.csv", "text/csv")

cursor.close()
conn.close()
