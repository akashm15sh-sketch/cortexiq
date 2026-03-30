import smtplib
import ssl
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_otp_email(receiver_email, otp_code):
    sender_email = os.getenv("GMAIL_USER")
    password = os.getenv("GMAIL_PASS")
    
    if not sender_email or not password:
        print("Email credentials not found in environment.")
        return False

    message = MIMEMultipart("alternative")
    message["Subject"] = f"CortexIQ OTP: {otp_code}"
    message["From"] = f"CortexIQ Neural Engine <{sender_email}>"
    message["To"] = receiver_email

    text = f"Your CortexIQ verification code is: {otp_code}\n\nThis code will expire in 5 minutes."
    html = f"""
    <html>
    <body>
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #1e293b; border-radius: 12px; background-color: #0f172a; color: #f8fafc;">
            <h2 style="color: #14b8a6;">CortexIQ Verification</h2>
            <p>Your verification code is:</p>
            <div style="font-size: 32px; font-weight: bold; letter-spacing: 5px; color: #14b8a6; padding: 10px 0;">
                {otp_code}
            </div>
            <p style="color: #64748b; font-size: 14px;">This code will expire in 5 minutes.</p>
            <hr style="border: 0; border-top: 1px solid #1e293b; margin: 20px 0;">
            <p style="font-size: 12px; color: #64748b;">If you did not request this, please ignore this email.</p>
        </div>
    </body>
    </html>
    """

    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    message.attach(part1)
    message.attach(part2)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
