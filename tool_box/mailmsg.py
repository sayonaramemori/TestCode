import smtplib
from email.message import EmailMessage

def send_over_msg():
# Email configuration
    sender_email = '13549683642@163.com'
    receiver_email = '1342733420@qq.com'
    subject = 'Reminder'
    body = 'Hello! Your training work is over now. Please check it as quickly as possible'

# Create the email message
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

# Send the email using Gmail's SMTP server
    with smtplib.SMTP_SSL('smtp.163.com', 465) as smtp:
        smtp.login(sender_email, 'MPg9BmhnVpHHApLT')
        smtp.send_message(msg)

if __name__ == '__main__':
    send_over_msg()
