SOUNDLENS V1 SAFE UPDATE

This package keeps the existing analysis, Artist Match, Stripe, account, report, and admin structures.

Added:
- Password reset request/confirmation flow
- Contact form and stored contact messages
- Professional Terms, Privacy, Refund pages
- Account deletion with password confirmation and automatic backup
- Admin data ZIP export
- Additional admin analytics (returning users, page views, clicks, analysis completion rate)
- Page, click, and session tracking
- Additional mobile layout polish

DATA SAFETY:
Do not replace your production soundlens_users.json, soundlens_admin_events.json, soundlens_feedback.json, soundlens_contact_messages.json, or saved_reports folder. Deploy the code files while preserving those data files. The Admin page now includes Download Data Backup.

SMTP is required for password-reset and contact notification emails. Existing SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL settings are used.

Before deployment:
1. Download an admin data backup.
2. Keep a copy of the current live deployment.
3. Replace code files only.
4. Verify login, upload, report, Artist Match, Stripe checkout, admin stats, reset email, contact, and account deletion using a test account.

Legal pages are professional working drafts and have not been reviewed by a lawyer.
