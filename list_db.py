import psycopg2
import json
from datetime import datetime

def list_database_content():
    """List all candidates in the database."""
    try:
        conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/cvscreening")
        cur = conn.cursor()

        # Get count
        cur.execute("SELECT COUNT(*) FROM candidates;")
        count = cur.fetchone()[0]
        
        print(f"\n📊 DATABASE STATUS")
        print(f"Total Candidates: {count}")
        print("-" * 50)

        if count > 0:
            # Get list of candidates
            cur.execute("""
                SELECT id, full_name, email, headline, total_experience_years, created_at
                FROM candidates 
                ORDER BY created_at DESC
            """)
            
            rows = cur.fetchall()
            
            print(f"{'ID':<38} | {'Name':<20} | {'Exp (Yrs)':<10} | {'Created At'}")
            print("-" * 90)
            
            for row in rows:
                c_id, name, email, headline, exp, created_at = row
                exp_str = f"{exp:.1f}" if exp else "N/A"
                print(f"{c_id:<38} | {name[:20]:<20} | {exp_str:<10} | {created_at.strftime('%Y-%m-%d %H:%M')}")
                
        print("-" * 50 + "\n")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")

if __name__ == "__main__":
    list_database_content()
