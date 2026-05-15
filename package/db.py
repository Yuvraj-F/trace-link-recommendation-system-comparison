import sqlite3

class SEOSS33:
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cursor = self.conn.cursor()

    def definition(self):
        return self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

    def table_definition(self, name):
        return self.cursor.execute(f"PRAGMA table_info({name});")

    def get_issues(self):
       return self.cursor.execute("SELECT * FROM issue;")
    
    def get_issue_links(self):
       return self.cursor.execute("SELECT * FROM issue_link;")
    
    def close(self):
        self.conn.close()
