import asyncio
import asyncpg

import os

# Direct connection to the Supabase Postgres instance, loaded dynamically from env
direct_url = os.getenv("DIRECT_DATABASE_URL") or os.getenv("DATABASE_URL") or "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=require"

async def migrate():
    print("Connecting directly to Supabase Postgres...")
    conn = await asyncpg.connect(direct_url)
    try:
        # Run DDL to add avatar_url column if not exists
        await conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;")
        print("Success: avatar_url column verified/added successfully!")
        
        # Check current columns of users table
        columns = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = 'users';")
        print("Current users table columns:", [col['column_name'] for col in columns])
    except Exception as e:
        print("Error running migration:", e)
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate())
