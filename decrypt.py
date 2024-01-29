import pandas as pd
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

# Contoh data admin
admin_data = [
    {"nama": "Admin1", "no_hp": "1234567890", "email": "1@1", "password": bcrypt.generate_password_hash("1").decode('utf-8')},
    {"nama": "Admin2", "no_hp": "9876543210", "email": "admin2@example.com", "password": bcrypt.generate_password_hash("password2").decode('utf-8')}
]

admin_df = pd.DataFrame(admin_data)
admin_df.to_csv('admin.csv', index=False)
