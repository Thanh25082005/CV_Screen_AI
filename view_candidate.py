import json
import psycopg2
from pprint import pprint
import sys

def view_candidate(name):
    try:
        conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/cvscreening")
        cur = conn.cursor()

        # Find the latest candidate by name
        cur.execute("""
            SELECT full_name, email, headline, total_experience_years, top_skills, raw_resume 
            FROM candidates 
            WHERE full_name ILIKE %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (f"%{name}%",))
        
        row = cur.fetchone()
        if not row:
            print(f"❌ Không tìm thấy ứng viên nào có tên: {name}")
            return

        full_name, email, headline, exp, skills, raw_resume = row

        print("\n" + "="*50)
        print(f"📄 THÔNG TIN ỨNG VIÊN: {full_name}")
        print("="*50)
        print(f"📧 Email: {email}")
        print(f"💼 Headline: {headline or 'N/A'}")
        print(f"⏳ Kinh nghiệm: {exp} năm")
        print(f"🛠️ Kỹ năng chính: {', '.join(skills[:10])}...")
        print("\n--- CHI TIẾT JSON ĐÃ PARSE (GROQ) ---")
        
        # Pretty print the raw resume JSON
        print(json.dumps(raw_resume, indent=2, ensure_ascii=False))
        
        print("="*50)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    search_name = sys.argv[1] if len(sys.argv) > 1 else "LE_KHANH LY"
    view_candidate(search_name)
