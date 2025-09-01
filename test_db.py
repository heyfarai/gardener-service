import asyncio
import aiosqlite
import os

async def check_db():
    db_path = "gardener.db"
    if os.path.exists(db_path):
        print(f"Database file exists at: {os.path.abspath(db_path)}")
        async with aiosqlite.connect(db_path) as db:
            # Check if tables exist
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = await cursor.fetchall()
            print("\nTables in database:")
            for table in tables:
                table_name = table[0]
                print(f"\nTable: {table_name}")
                try:
                    # Get table info
                    cursor = await db.execute(f"PRAGMA table_info({table_name})")
                    columns = await cursor.fetchall()
                    print("Columns:")
                    for col in columns:
                        print(f"  {col[1]} ({col[2]})")
                except Exception as e:
                    print(f"  Error getting schema: {e}")
    else:
        print(f"Database file not found at: {os.path.abspath(db_path)}")

if __name__ == "__main__":
    asyncio.run(check_db())
