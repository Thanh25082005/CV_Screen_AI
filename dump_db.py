
import psycopg2
import json

def dump_db():
    try:
        conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/cvscreening")
        cur = conn.cursor()
        
        # Get Candidates
        cur.execute("SELECT id, full_name, email, phone, headline, total_experience_years, top_skills, summary FROM candidates")
        candidates = cur.fetchall()
        
        print(f"\n✅ Found {len(candidates)} Candidates in DB:")
        print("="*60)
        
        for c in candidates:
            c_id, name, email, phone, headline, exp, skills, summary = c
            print(f"🆔 ID: {c_id}")
            print(f"👤 Name: {name}")
            print(f"📧 Email: {email}")
            print(f"📱 Phone: {phone}")
            print(f"💼 Headline: {headline}")
            print(f"⭐ Experience: {exp} years")
            print(f"🛠️ Skills: {json.dumps(skills, ensure_ascii=False)}")
            print(f"📝 Summary: {summary[:100]}..." if summary else "📝 Summary: N/A")
            
            # Count chunks
            cur.execute("SELECT COUNT(*), section FROM chunks WHERE candidate_id = %s GROUP BY section", (c_id,))
            chunks = cur.fetchall()
            print("📚 Chunks breakdown:")
            for count, section in chunks:
                print(f"   - {section}: {count} chunks")
            print("-" * 60)

        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    dump_db()
