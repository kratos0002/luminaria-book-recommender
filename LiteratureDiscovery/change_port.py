"""
Simple script to change the port in app.py from 5001 to 5002.
"""

def change_port():
    with open('app.py', 'r') as f:
        content = f.read()
    
    # Change port from 5001 to 5002
    content = content.replace('port=5001', 'port=5002')
    
    with open('app.py', 'w') as f:
        f.write(content)
    
    print("Changed port from 5001 to 5002 in app.py")

if __name__ == "__main__":
    change_port()
