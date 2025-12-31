#for reset email template
from datetime import datetime
def email_verification_body(user_email,verify_url):
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Verify Your Email</title>
</head>
<body style="margin:0; padding:0; background-color:#0D0D0D; font-family:Arial, sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0">
    <tr>
      <td align="center" style="padding:40px 0;">
        
        <!-- Card -->
        <table width="600" cellpadding="0" cellspacing="0" style="background:#151515; border-radius:12px; padding:30px;">
          
          <!-- Logo -->
          <tr>
            <td align="center" style="padding-bottom:20px;">
              <h1 style="color:#FF7B00; margin:0;">IntelliDocs</h1>
            </td>
          </tr>

          <!-- Title -->
          <tr>
            <td align="center" style="color:#FFFFFF; font-size:22px; font-weight:bold;">
              Verify Your Email Address
            </td>
          </tr>

          <!-- Message -->
          <tr>
            <td style="color:#CCCCCC; font-size:15px; padding:20px 0; line-height:1.6;">
              Hi <b>{user_email}</b>,<br><br>

              Welcome to <b>IntelliDocs</b> 🎉<br>
              Please confirm your email address by clicking the button below.
              This verification link will expire in <b>10 minutes</b>.
            </td>
          </tr>

          <!-- Button -->
          <tr>
            <td align="center" style="padding:30px 0;">
              <a href="{verify_url}"
                 style="
                   background: linear-gradient(135deg, #FF7B00, #FF5100);
                   color:#FFFFFF;
                   text-decoration:none;
                   padding:14px 32px;
                   border-radius:30px;
                   font-weight:bold;
                   display:inline-block;
                 ">
                ✅ Verify Email
              </a>
            </td>
          </tr>

          <!-- Fallback -->
          <tr>
            <td style="color:#888888; font-size:13px; line-height:1.6;">
              If the button doesn’t work, copy and paste this link into your browser:<br>
              <span style="color:#FF7B00;">{verify_url}</span>
            </td>
          </tr>

          <!-- Footer -->
          <tr>
            <td style="color:#666666; font-size:12px; padding-top:30px; text-align:center;">
              If you didn’t create this account, you can safely ignore this email.<br><br>
              © {datetime.now().year} IntelliDocs. All rights reserved.
            </td>
          </tr>

        </table>

      </td>
    </tr>
  </table>

</body>
</html>
"""


def email_reset_password_body(user_email,reset_url):
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />            
  <title>Reset Your Password</title>
</head>
<body style="margin:0; padding:0; background-color:#0D0D0
D; font-family:Arial, sans-serif;">

  <table width="100%" cellpadding="0" cellspacing="0">
    <tr>
      <td align="center" style="padding:40px 0;">
        
        <!-- Card -->
        <table width="600" cellpadding="0" cellspacing="0" style="background:#151515; border-radius:12px; padding:30px;">
          
          <!-- Logo -->
          <tr>
            <td align="center" style="padding-bottom:20px;">
              <h1 style="color:#FF7B00; margin:0;">IntelliDocs</h1>
            </td>
          </tr>

          <!-- Title -->
          <tr>
            <td align="center" style="color:#FFFFFF; font-size:22px; font-weight:bold;">
              Reset Your Password
            </td>
          </tr>

          <!-- Message -->
          <tr>
            <td style="color:#CCCCCC; font-size:15px; padding:20px 0; line-height:1.6;">
              Hi <b>{user_email}</b>,<br><br>

              We received a request to reset your password.<br>
              Click the button below to proceed. This link will expire in <b>10 minutes</b>.
            </td>
          </tr>

          <!-- Button -->
          <tr>
            <td align="center" style="padding:30px 0;">
              <a href="{reset_url}"
                 style="
                   background: linear-gradient(135deg, #FF7B00, #FF5100);
                   color:#FFFFFF;
                   text-decoration:none;
                   padding:14px 32px;
                   border-radius:30px;
                   font-weight:bold;
                   display:inline-block;
                 ">
                🔒 Reset Password         
              </a>
            </td>
          </tr> 

          
          <!-- Fallback -->
          <tr>
            <td style="color:#888888; font-size:13px; line-height:1.6;">
              If the button doesn’t work, copy and paste this link into your browser:<br>
              <span style="color:#FF7B00;">{reset_url}</span>
            </td>
          </tr> 

          <!-- Footer -->
          <tr>
            <td style="color:#666666; font-size:12px; padding-top:
30px; text-align:center;">
              If you didn’t request a password reset, you can safely ignore this email.<br><br>
              © {datetime.now().year} IntelliDocs. All rights reserved.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""
