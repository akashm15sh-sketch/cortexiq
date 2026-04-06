"""Email OTP sender — uses Resend HTTP API (primary) or Gmail SMTP (fallback).

Resend: Free 3,000 emails/month — works on all cloud platforms (HTTP-based).
Gmail SMTP: Works locally but blocked by Render/Vercel/Railway free tiers.
"""
import os
import ssl
import smtplib
import httpx
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ── Branded HTML template ──
def _otp_html(otp_code: str) -> str:
    return f"""
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


def send_otp_email(receiver_email: str, otp_code: str) -> bool:
    """Send OTP email. Tries Resend HTTP API first, then Gmail SMTP fallback."""

    # ── Method 1: Resend HTTP API (works on all cloud platforms) ──
    resend_key = os.getenv("RESEND_API_KEY")
    if resend_key:
        try:
            result = _send_via_resend(resend_key, receiver_email, otp_code)
            if result:
                return True
            print("Resend API failed, trying Gmail SMTP fallback...")
        except Exception as e:
            print(f"Resend error: {e}, trying Gmail SMTP fallback...")

    # ── Method 2: Gmail SMTP (works locally, blocked on most cloud free tiers) ──
    gmail_user = os.getenv("GMAIL_USER")
    gmail_pass = os.getenv("GMAIL_PASS")
    if gmail_user and gmail_pass:
        try:
            result = _send_via_gmail(gmail_user, gmail_pass, receiver_email, otp_code)
            if result:
                return True
        except Exception as e:
            print(f"Error sending email: {e}")

    # ── Both methods failed ──
    print("All email methods failed. OTP will be shown on screen as fallback.")
    return False


def _send_via_resend(api_key: str, receiver_email: str, otp_code: str) -> bool:
    """Send email via Resend HTTP API (free: 3,000 emails/month)."""
    response = httpx.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": "CortexIQ <onboarding@resend.dev>",
            "to": [receiver_email],
            "subject": f"CortexIQ OTP: {otp_code}",
            "html": _otp_html(otp_code),
        },
        timeout=10,
    )
    if response.status_code in (200, 201):
        print(f"OTP sent via Resend to {receiver_email}")
        return True
    else:
        print(f"Resend API error: {response.status_code} — {response.text}")
        return False


def _send_via_gmail(sender_email: str, password: str, receiver_email: str, otp_code: str) -> bool:
    """Send email via Gmail SMTP (local dev / non-restricted hosts)."""
    message = MIMEMultipart("alternative")
    message["Subject"] = f"CortexIQ OTP: {otp_code}"
    message["From"] = f"CortexIQ Neural Engine <{sender_email}>"
    message["To"] = receiver_email

    text = f"Your CortexIQ verification code is: {otp_code}\n\nThis code will expire in 5 minutes."
    message.attach(MIMEText(text, "plain"))
    message.attach(MIMEText(_otp_html(otp_code), "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    print(f"OTP sent via Gmail to {receiver_email}")
    return True
