import psycopg2
import os
from dotenv import load_dotenv

def clear_database():
    """Manual script to clear all candidates and chunks from PostgreSQL."""
    load_dotenv()
    
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/cvscreening")
    
    print(f"⚠️  DANGER: About to clear all data from {db_url}")
    confirm = input("Are you sure? (y/N): ")
    
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # Delete chunks first (though cascade usually handles it, being explicit is safer in raw SQL)
        print("Clearing chunks table...")
        cur.execute("DELETE FROM chunks;")
        
        print("Clearing candidates table...")
        cur.execute("DELETE FROM candidates;")

        conn.commit()
        print("✅ Database cleared successfully!")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    clear_database()
