# LUMO — ISSUE PLAYBOOKS
**Version 1.0 | Internal Support Knowledge Base**

---

## Overview

This document contains step-by-step resolution playbooks for every issue type agents handle. Each playbook defines: what to ask, what to check, what the resolution paths are, what policy applies, and when to escalate. Follow playbooks in order. Do not skip steps — each step is there because it has resolved or clarified the issue in past cases.

**How to use this document:**
- Find the issue type matching the customer's problem
- Follow the steps in sequence
- Each playbook ends with escalation criteria — if those conditions are met, escalate before attempting further resolution
- After resolution, confirm with the customer that the issue is resolved before closing the ticket

---

## PLAYBOOK 1: Account Login & Access Issues

### 1A: Forgotten Email Address

**When to use:** Customer says they can't log in and doesn't know which email they used to sign up for Lumo.

**Step 1 — Gather information:**
Ask the customer:
- What is the name or URL of their Lumo workspace? (e.g., lumo.com/companyname)
- Do they remember roughly when they joined?
- What email domains might they have used? (personal, work, old work?)

**Step 2 — Explain the lookup limitation:**
Lumo cannot look up accounts by name — only by email address and workspace. Tell the customer:
> "I'm unable to search for accounts by name. To help you, I'll need to identify which email address is associated with your account."

**Step 3 — Resolution path A — Admin lookup:**
If the customer knows their workspace and there is an admin available:
- Ask their admin to go to Settings > Administration > Members
- Admin can see the email address associated with every member
- Admin can update the email if the wrong one was used

**Step 4 — Resolution path B — Self-service guess and send:**
If no admin is available:
- Ask customer to try requesting a magic link at lumo.com/signin for each email address they might have used
- If the email exists in Lumo, they receive a sign-in link
- If not, they see "No account found" — try the next address

**Step 5 — Resolution path C — Workspace URL clue:**
If the customer knows the workspace URL but not their email:
- They can attempt to sign in at that workspace URL
- The sign-in page for known workspaces sometimes shows a masked email hint (e.g., d***@company.com) if they have previously signed in from that browser

**Escalation criteria:** Escalate if customer is the Workspace Owner and cannot access their account and there are no other admins — this requires identity verification and manual support intervention.

---

### 1B: Password Reset Email Not Received

**When to use:** Customer requested a password reset but the email has not arrived.

**Step 1 — Gather information:**
Ask:
- What email address did they request the reset for?
- What is their workspace name or URL?
- Does their organization use SSO (Single Sign-On) such as Google, Okta, or Microsoft?

**Step 2 — Check for SSO:**
This is the most common cause of this issue. If the customer's organization is on Business+ and uses SAML SSO:
- Password reset does NOT go through Lumo — it goes through their identity provider
- Tell them: "If your organization uses Okta, Microsoft Entra, or Google SSO, your password needs to be reset through that system, not through Lumo. Please contact your IT team."
- Signs the org uses SSO: login page redirects to a company-branded page instead of Lumo's standard login

**Step 3 — Check spam and filters:**
If not on SSO:
- Ask customer to check spam, junk, promotions, and "other" folders
- Ask if their corporate email has security filters that might strip links from external senders
- Lumo's password reset emails come from: no-reply@lumo.com
- Ask IT to whitelist no-reply@lumo.com if filtering is the issue

**Step 4 — Resend the reset email:**
- Confirm the exact email address and ask customer to request a new reset
- Lumo allows up to 3 resets per hour — if they've already tried 3 times in the last hour, ask them to wait 1 hour before requesting again
- Magic links are valid for 30 minutes — if the customer clicked the link late, they need a fresh one

**Step 5 — Verify email address:**
- If the reset email consistently doesn't arrive: it's possible the email they're entering is not the one on their Lumo account
- Refer back to Playbook 1A to identify the correct email address

**Escalation criteria:** If customer has confirmed: correct email, not on SSO, not in spam, waited 1 hour and tried again — escalate for engineering to check email delivery logs.

---

### 1C: Account Locked After Too Many Login Attempts

**When to use:** Customer says they've tried logging in multiple times and are now getting an error preventing further attempts.

**Step 1 — Clarify the error:**
Ask the customer to describe the exact error message. Lumo does not permanently lock accounts, but browsers and apps may behave differently:
- "Too many attempts" in browser: usually a browser-level or Cloudflare rate limit — resolve by clearing browser cache or using incognito/private mode
- "Account suspended": this would be an admin action — see Step 3
- "Invalid email or password": standard wrong credentials, not a lockout

**Step 2 — Browser/app troubleshoot:**
- Ask customer to try in a private/incognito browser window
- Try a different browser entirely (Chrome recommended)
- Clear browser cache and cookies for lumo.com
- Try the Lumo mobile app as an alternative while resolving desktop issue

**Step 3 — Check if deactivated:**
If the customer's account shows as deactivated:
- A workspace admin has manually deactivated them
- Only an admin can reactivate — direct customer to their IT or HR department
- Support cannot reactivate a member without admin authorization

**Resolution:** In most cases, incognito window resolves the issue. If admin deactivation is confirmed, the resolution is with their company's admin, not Lumo Support.

---

### 1D: New Device — Can't Receive Verification Code

**When to use:** Customer is on a new device and Lumo is asking for a verification code they can't receive.

**Step 1 — Identify the verification type:**
- Is this an email magic link for sign-in? — follow Playbook 1B
- Is this a 2FA code? — follow Playbook 2A or 2C

**Step 2 — For email verification on new device:**
- The magic link is sent to the email on their Lumo account
- Common issue: corporate email security may strip or delay links from external senders
- Ask customer to try requesting the link again and clicking it within 5 minutes
- If email is consistently not received: ask IT to whitelist no-reply@lumo.com

**Resolution:** Resend the verification email and have the customer act on it promptly (within 5 minutes to avoid expiry).

---

## PLAYBOOK 2: Two-Factor Authentication (2FA) Issues

### 2A: Member Locked Out Due to Lost/Changed Authenticator

**When to use:** A regular workspace member (not admin/owner) cannot log in because they lost their phone, got a new phone, or their authenticator app isn't generating correct codes.

**Step 1 — Confirm they are not the only admin:**
Ask: "Are you a workspace admin or owner, or a regular member?"
- If regular member: proceed to Step 2
- If admin/owner: see Playbook 2B

**Step 2 — Direct to workspace admin:**
The correct resolution for a regular member is to have their workspace admin disable 2FA for their account:
- Admin path: Settings > Administration > Members > [member name] > Security > Disable 2FA
- Once disabled, the member can log in without 2FA and set it up fresh on their new device
- This does not require Lumo Support involvement — it's a self-service admin action

**Step 3 — If admin is unavailable:**
If the member cannot reach their admin (e.g., admin is on vacation, admin has also left the company):
- Lumo Support can disable 2FA for the member after identity verification
- Verification required: email on Lumo account + workspace URL + confirmation from any admin (even via email to Support)
- Resolution time: within 1 business day

**Resolution:** Workspace admin disables 2FA for the affected member. Member logs in and sets up fresh 2FA. Always recommend setting up backup codes or a backup authenticator device to prevent this in future.

---

### 2B: Admin or Owner Locked Out Due to 2FA

**When to use:** The workspace admin or owner cannot log in because of 2FA issues and there is no other admin to help them.

**This is a Support-required escalation — do not attempt to resolve without following verification steps.**

**Step 1 — Verify identity:**
Before making any changes, collect and verify:
1. Billing email address on file for the workspace
2. Last 4 digits of the payment card on file
3. Workspace URL (lumo.com/workspace-name)
4. For Business+/Enterprise: additionally, the account manager may be asked to confirm

All three items must match exactly. If the customer cannot provide all three, do not proceed.

**Step 2 — Record verification in ticket:**
Note in the ticket: "Identity verified via billing email [email], card last4 [xxxx], workspace [url] on [date] by agent [name]."

**Step 3 — Disable 2FA and restore access:**
After successful verification:
- Disable 2FA for the specific admin/owner account via backend tooling
- Send the customer a magic link to sign in without 2FA
- Ask them to set up fresh 2FA immediately upon signing in
- Recommend adding a backup admin account that does not use 2FA as a recovery option going forward

**Resolution time:** Same business day for Business+ and Enterprise Grid. Within 1 business day for Pro.

**Escalation criteria:** Escalate to senior agent immediately if: customer cannot provide verification information, if this involves an Enterprise Grid workspace, or if there are signs of suspicious activity (someone other than the legitimate owner trying to gain access).

---

### 2C: SMS 2FA Codes Not Being Received

**When to use:** Customer has SMS-based 2FA enabled but text messages with codes are not arriving.

**Step 1 — Check phone number format:**
- Ask the customer to confirm their phone number is saved in Lumo with full country code format (e.g., +1 415 555 0123 for US)
- Numbers without country codes often fail for international users

**Step 2 — Check carrier restrictions:**
- Some mobile carriers block automated SMS from short codes (common in certain countries)
- Ask customer: are they receiving other SMS messages fine? If not, this is a carrier issue
- International SMS delivery can be delayed by 5-10 minutes in some regions

**Step 3 — Recommend switching to authenticator app:**
SMS 2FA is inherently less reliable than authenticator apps. Recommend:
> "For more reliable 2FA, I'd recommend switching to an authenticator app like Google Authenticator, Authy, or 1Password. These work offline and don't rely on mobile network delivery."
- If they have access to their account (logged in on another device), direct to Preferences > Security > Two-Factor Authentication to add an authenticator app and remove SMS

**Step 4 — Temporary bypass if completely locked out:**
If the customer is completely locked out (no other devices logged in):
- Follow Playbook 2A or 2B depending on their role
- After regaining access, immediately switch from SMS to authenticator app

---

### 2D: Organization-Wide 2FA Enforcement Issue

**When to use:** Admin has enabled mandatory 2FA for all workspace members, and a member is unable to set it up.

**Step 1 — Identify the specific issue:**
Why can't they set up 2FA?
- Don't have a smartphone: use desktop-based authenticator (Bitwarden, 1Password desktop app)
- Authenticator app was deleted: can reinstall and set up fresh (need to generate new QR code from Lumo settings)
- QR code not scanning: try entering the setup key manually instead of scanning

**Step 2 — Temporary admin workaround:**
If the member genuinely cannot set up 2FA right now (e.g., IT hasn't approved an authenticator app):
- Admin can temporarily disable org-wide 2FA enforcement (Settings > Authentication > Two-Factor Authentication)
- Member gets set up, then admin re-enables enforcement
- This is a brief, controlled window — should not be left disabled for more than a few hours

**Resolution:** Most cases resolve by assisting the member through authenticator app setup or switching from QR scan to manual key entry.

---

## PLAYBOOK 3: Billing & Charge Issues

### 3A: Unexpected Charge — Customer Doesn't Know What It's For

**When to use:** Customer contacts support saying a charge appeared from Lumo that they don't understand.

**Step 1 — Gather information:**
Ask:
- What is the workspace name or URL associated with the charge?
- What is the approximate amount and date of the charge?
- What is the billing email address on the account?
- Are they the workspace owner or admin, or just the cardholder?

**Step 2 — Identify the charge type:**
Common charge types and their causes:

| Charge Type | How to identify | Explanation to give customer |
|---|---|---|
| Regular monthly/annual renewal | Amount matches plan × seats | "This is your regular Lumo subscription renewal for [X] seats on [plan]." |
| Seat addition proration | Mid-month, partial amount | "This charge occurred when [X] seats were added to your workspace on [date]. It covers the pro-rated cost for the rest of the billing period." |
| Plan upgrade proration | Mid-month, partial amount | "When your workspace was upgraded from [old plan] to [new plan] on [date], this charge covered the cost difference for the remaining days in your billing period." |
| Annual renewal (forgotten) | Large lump sum, once per year | "Your Lumo workspace is on annual billing. This charge is your annual renewal covering [date range]." |
| Lumo Connect charge | Small amount, labeled 'Connect' | "This charge is related to Lumo Connect usage on your workspace." |

**Step 3 — Pull up the invoice:**
- Ask customer to download the invoice from Settings > Administration > Billing > Invoice History
- Walk through each line item with them
- If they cannot access billing settings: offer to resend the invoice to their billing email on file

**Step 4 — If the charge is incorrect:**
- If it's a duplicate charge: follow Playbook 3D
- If the amount seems wrong for their plan and seat count: verify the seat count in their billing settings matches what they expect
- If charge occurred after cancellation: follow Playbook 3E

**Resolution:** Most unexpected charges are explained once the invoice is reviewed. Confirm with customer that the explanation makes sense before closing.

---

### 3B: Refund Request — Within 30-Day Window

**When to use:** Customer is requesting a refund and the charge was within the last 30 days.

**Step 1 — Verify eligibility:**
Confirm ALL of the following:
- Charge date is within 30 calendar days of today
- This is the first time this workspace has been on a paid plan
- The plan is Pro or Business+ (not Enterprise Grid — that has separate refund handling)
- The request is coming from the workspace owner or billing admin

If any condition is not met, this does not qualify as a standard refund. See Playbook 3C for out-of-window cases.

**Step 2 — Confirm the refund details:**
- Exact charge amount to refund
- Billing email on the account
- Last 4 digits of the card (for verification and to confirm which card receives the refund)

**Step 3 — Process the refund:**
- Process via billing system
- Note in ticket: amount, date processed, agent name, authorization basis
- Inform customer: "Your refund of [amount] has been processed to the card ending in [last4]. It will appear on your statement within 5–7 business days."

**Step 4 — Confirm cancellation if applicable:**
If the customer is requesting a refund because they want to stop using Lumo:
- Ask if they would also like to cancel the workspace
- If yes: walk them through Settings > Administration > Billing > Cancel Plan
- If no: leave workspace active (some customers want a refund but plan to continue using Free tier)

**Resolution:** Refund processed, customer informed of timeline, workspace status confirmed.

---

### 3C: Refund Request — Outside 30-Day Window or Repeat Subscription

**When to use:** Customer is requesting a refund but the standard 30-day window has passed OR this is a renewal charge (not first payment).

**Step 1 — Acknowledge the request empathetically:**
Do not lead with "that's not our policy." Start with:
> "I understand you'd like a refund for this charge. Let me look into your account and see what options are available."

**Step 2 — Check for courtesy refund eligibility:**
Review the specific circumstances. Courtesy refunds outside the standard window are possible only if one of these applies:
- Accidental workspace creation (unused workspace, within 14 days of discovery)
- Documented billing error (wrong amount, system error)
- Charge occurred after a confirmed cancellation
- Unauthorized account creation by an employee (with supporting documentation)

If none apply: the refund is not eligible under policy. Skip to Step 4.

**Step 3 — If courtesy refund applies:**
- Note the specific justification clearly in the ticket
- Escalate to senior agent for approval before processing
- Do NOT tell the customer a refund is approved until senior agent has authorized it

**Step 4 — If refund is not eligible:**
Explain clearly and empathetically:
> "Our refund policy covers the first 30 days of a new subscription. Since this charge is [X days/months] past that window, I'm not able to issue a refund for this specific charge. What I can do is [offer alternatives — see Step 5]."

**Step 5 — Offer alternatives:**
Even when a refund isn't possible, offer something:
- If on annual plan: ensure they know they keep full access for the remainder of the year they paid
- Offer to downgrade their plan at next renewal to reduce future costs
- If cost is the concern: check if they qualify for nonprofit/education discount (85% off)
- If switching to a competitor: acknowledge, don't be defensive, wish them well — a good exit experience is better for the brand than a hostile one

**Escalation criteria:** Any refund outside the standard window requires senior agent approval before processing.

---

### 3D: Duplicate Charge

**When to use:** Customer has been charged twice for the same billing period.

**Step 1 — Confirm the duplicate:**
Ask for:
- The two charge dates and amounts
- Last 4 digits of the card
- Workspace name

Check billing system to verify that two charges exist for the same period. Do not process a refund based solely on the customer's claim — verify in the system first.

**Step 2 — Confirm it is actually a duplicate:**
Common non-duplicate situations that look like duplicates:
- Upgrade proration + regular renewal on same billing date (these are two different charges for different things — not a duplicate)
- Seat addition proration + regular renewal (same — not a duplicate)
- Two different workspaces belonging to the same cardholder (different workspaces, not a duplicate)

A true duplicate is: two charges for the exact same amount, same workspace, same billing period, with no corresponding event (upgrade, seat addition) that would explain both.

**Step 3 — Process the refund:**
If confirmed as a true duplicate:
- Refund the duplicate charge immediately — no senior approval needed
- Timeline: 5–7 business days
- Note in ticket: confirmed duplicate, system error flagged for engineering review

**Resolution:** Refund of duplicate charge processed. Flag for engineering if it appears to be a system error.

---

### 3E: Charged After Cancellation

**When to use:** Customer says they cancelled their Lumo subscription but were still charged afterward.

**Step 1 — Determine the cancellation date:**
Ask for the date they cancelled and whether they received a cancellation confirmation email.

**Step 2 — Check the cancellation record:**
In the billing system:
- Was the cancellation processed before the charge date? If yes → full refund, no questions asked
- Was the cancellation submitted but never finalized? (Customer started cancellation flow but didn't confirm final step) → this is not a processed cancellation
- Is there no cancellation record? → customer may believe they cancelled but didn't complete the process

**Step 3 — Resolution based on findings:**

If cancellation was confirmed before charge date:
- Issue full refund immediately
- Apologize for the error
- Confirm workspace is now cancelled and access will end [date]

If cancellation was not properly completed:
- Explain kindly: "I can see a cancellation was started but not finalized. The account remained active and that's why the charge went through."
- Cannot refund standard renewal charges when cancellation wasn't completed
- However: if customer is insistent and this was clearly unintentional, escalate for senior review — a goodwill gesture may be appropriate

If no cancellation record exists:
- Walk customer through how to cancel: Settings > Administration > Billing > Cancel Plan
- Explain the charge was for the active subscription
- Standard refund policy applies (within 30 days for first charge only)

---

### 3F: Proration Charge Confusion

**When to use:** Customer is confused about a mid-cycle charge that appeared after they added seats or changed plans.

**Step 1 — Explain proration:**
> "When seats are added or a plan is upgraded mid-billing cycle, Lumo charges a prorated amount for the remainder of the current period. This ensures you're only charged for the time the new seats or plan features were active."

**Step 2 — Walk through the math:**
Calculate with the customer:
- Days remaining in billing period when change was made
- New daily rate vs old daily rate
- Proration = (new rate - old rate) × days remaining ÷ days in billing period

Example: 10 seats added on day 20 of a 30-day month at $7.25/seat/month:
- Daily rate per new seat = $7.25 / 30 = $0.2417
- Proration = 10 seats × $0.2417 × 10 remaining days = $24.17

**Step 3 — If calculation seems wrong:**
If the customer believes the proration amount is incorrect after going through the math:
- Note the discrepancy in the ticket
- Escalate to senior agent to review the billing calculation
- Do not confirm or deny the error until reviewed

---

## PLAYBOOK 4: Plan Changes

### 4A: How to Upgrade

**When to use:** Customer wants to upgrade their plan or add more seats.

**Step 1 — Confirm who is trying to upgrade:**
Only workspace Owners and Admins with billing access can upgrade. If a regular member is asking:
> "Only workspace admins can change the plan. Please ask your workspace owner or an admin to initiate the upgrade."

**Step 2 — Walk through the upgrade:**
Path: Settings > Administration > Billing > Upgrade Plan
- Select new plan
- Review the proration charge (shown before confirming)
- Enter or confirm payment method
- Confirm upgrade

**Step 3 — Explain what happens immediately:**
- Upgrade takes effect immediately after confirmation
- Features of new plan are available right away
- A prorated charge for the remainder of the current billing cycle appears on their card
- Next regular billing date: charged at the new plan rate for the full period

**Step 4 — Explain the 30-day refund window:**
For first-time upgraders:
> "Just so you know, if you decide within 30 days that Business+ isn't right for you, we do offer a refund on that first charge."

---

### 4B: How to Downgrade

**When to use:** Customer wants to reduce their plan tier.

**Step 1 — Understand the reason:**
Ask: "What's prompting the change? I want to make sure the new plan will meet your needs."
Common reasons and responses:
- Cost: check if they qualify for discounts, offer annual billing savings
- Don't use a specific feature: identify the feature and confirm they won't need it
- Team size reduction: confirm how many seats they actually need

**Step 2 — Warn about feature loss:**
Before confirming the downgrade, clearly explain what they will lose. Be specific:

For Business+ → Pro:
> "Downgrading to Pro means you'll lose: SAML SSO (you'll need to use Google SSO or email login instead), the ability to export DMs and private channels for compliance, advanced AI features including AI search and daily recaps, and the 99.99% uptime SLA."

For Pro → Free:
> "Important: Downgrading to Free means messages older than 90 days will become hidden — they're not deleted, but they won't be visible or searchable until you re-upgrade. You'll also be limited to 10 app integrations."

**Step 3 — Confirm understanding and proceed:**
Ask: "Are you comfortable with those changes? Would you like to go ahead with the downgrade?"
If yes: walk them through Settings > Administration > Billing > Change Plan > select new plan
Note: downgrade takes effect at next billing cycle renewal — they keep current features until then.

**Step 4 — Seat count reduction:**
Remind the customer: deactivating members does NOT automatically reduce the seat count in billing. They must also:
- Go to Settings > Administration > Billing > Manage Seats
- Reduce the seat count to their desired number
- This takes effect at next billing cycle

---

### 4C: How to Cancel

**When to use:** Customer wants to cancel their Lumo subscription entirely.

**Step 1 — Understand the reason and attempt retention:**
Ask: "I'm sorry to hear you're considering leaving. Could you tell me what's driving this decision?"
Common scenarios and retention responses:
- Too expensive: "I understand. Have you considered [annual billing / downgrading to Pro / Free plan]? That might meet your needs at a much lower cost."
- Switching to Teams/Google Chat: "I understand. Many teams find Lumo's [specific feature they use] hard to replace — is there anything specific that's frustrating you that I might be able to help with?"
- Company shutting down: "I'm sorry to hear that. I can help make sure your data is exported before you close the workspace." — do not attempt retention in this case.

**Step 2 — Advise on data export before cancelling:**
This is critical and often overlooked:
> "Before you cancel, I'd strongly recommend exporting your workspace data. Once the workspace reverts to Free plan, messages older than 90 days start to become hidden."

Export capability by plan:
- Business+: can export everything (all channels, DMs, files) from Settings > Administration > Workspace Settings > Import/Export Data
- Pro/Free: can only export public channel data
- If they need full export but are only on Pro: they would need to briefly upgrade to Business+ to complete the export — explain this honestly

**Step 3 — Walk through cancellation:**
Settings > Administration > Billing > Cancel Plan
After cancellation is confirmed:
- Workspace remains on current paid plan until end of current billing period
- After that: workspace reverts to Free plan (not deleted)
- Customer has access to all current data until the Free plan limits apply

**Step 4 — Confirm cancellation is complete:**
Ask customer to confirm they received the cancellation confirmation email. If they did not receive it: check the cancellation was successfully processed in the system.

---

## PLAYBOOK 5: SSO & Authentication Issues

### 5A: SAML SSO Not Working

**When to use:** Customer has SAML SSO configured (Business+ or Enterprise Grid) and it has stopped working or is not working for some users.

**Step 1 — Identify the specific error:**
Ask customer for the exact error message. Common errors and their causes:

**"SAML response is invalid"**
- Most common cause: the SAML certificate in the identity provider (IdP) has expired
- Fix: renew the certificate in Okta/Entra/OneLogin and re-upload the metadata XML to Lumo
- Path in Lumo: Settings > Authentication > SAML Configuration > Update Metadata

**"User not found" after SSO login**
- Most common cause: the email address in the IdP profile does not match the email in Lumo
- Fix: ensure the IdP is passing the email attribute that matches the Lumo account
- Check: what email attribute mapping is configured in the IdP application settings

**Infinite redirect loop**
- Most common cause: the ACS (Assertion Consumer Service) URL in the IdP doesn't match what Lumo expects
- Fix: copy the exact ACS URL from Lumo's SAML configuration page and paste it into the IdP
- Path in Lumo: Settings > Authentication > SAML Configuration > Copy ACS URL

**"SSO works for some users but not others"**
- Most common cause: those users are not assigned to the Lumo application in the IdP
- Fix: in the IdP, check the application assignment and add the affected users or their group

**Step 2 — Emergency SSO disable:**
If SSO is misconfigured and ALL users are locked out (no one can log in):
- Verify identity of the workspace owner (see standard verification: billing email + card last4 + workspace URL)
- Support can disable SSO via backend for Business+ and Enterprise Grid
- After disabling: users can log in with email/password or Google SSO
- Admin should fix the SSO configuration before re-enabling

**Step 3 — Escalation:**
If the error message doesn't match any of the above patterns and basic troubleshooting hasn't resolved it:
- Collect: IdP type (Okta/Entra/etc.), exact error message, screenshot if possible, metadata XML if they can share it
- Escalate to senior agent with all collected information

---

### 5B: Google SSO Not Working

**When to use:** Customer is trying to sign in with Google and it's not working.

**Step 1 — Check if their Google Workspace admin has restricted third-party apps:**
This is the most common cause. If a Google Workspace admin has restricted which third-party apps can request access, Lumo sign-in via Google may be blocked.
- Resolution: customer needs to ask their Google Workspace admin to allow Lumo in the list of permitted apps (Google Admin Console > Security > API Controls > App Access Control)

**Step 2 — Check if they're using the correct Google account:**
- If the customer has multiple Google accounts, they may be trying to sign in with a personal Gmail instead of their work Google account
- Ask: "Are you trying to sign in with your work Google account (e.g., you@company.com) or a personal Gmail?"

**Step 3 — Browser extension interference:**
- Some browser extensions (ad blockers, privacy tools) can interfere with the Google OAuth popup
- Ask customer to try in incognito mode (extensions typically disabled) or a different browser

---

## PLAYBOOK 6: Technical Performance Issues

### 6A: Lumo App Running Slowly

**When to use:** Customer reports the Lumo desktop or mobile app is slow, laggy, or taking a long time to load.

**Step 1 — Check Lumo's status page:**
Before troubleshooting: check status.lumo.com for any active incidents.
- If there is an active incident: "I can see there's a known issue affecting performance right now. Our engineering team is working on it. Here's the current status: [status page link]. I'd recommend checking back in [estimated resolution time]."
- If no incident: proceed to Step 2

**Step 2 — Identify the platform and scope:**
Ask:
- Desktop app, mobile app, or browser?
- What operating system and version?
- What version of the Lumo app are they on?
- Is it slow for everyone on their team, or just them?
- Did it start after a specific event (app update, OS update, network change)?

**Step 3 — Platform-specific troubleshooting:**

Desktop app:
- Update to latest Lumo version (Help > Check for Updates)
- Quit and relaunch the app completely
- Try using Lumo in a browser (Chrome at lumo.com) to isolate whether it's the app or the service
- Clear Lumo's local cache: quit app > delete cache folder (location varies by OS) > relaunch
- Disable hardware acceleration: Settings > Advanced > Disable hardware acceleration

Browser (Chrome recommended):
- Clear browser cache and cookies for lumo.com
- Disable browser extensions (try incognito mode)
- Check if the issue is specific to one workspace or all Lumo workspaces

VPN or corporate network:
- Many performance issues are caused by VPN routing adding latency
- Ask customer to try on a non-VPN connection to isolate
- If corporate firewall is blocking Lumo's domains: IT needs to whitelist *.lumo.com and *.lumo-edge.com

**Step 4 — Specific channel loading issue:**
If a specific channel loads slowly but others are fine:
- Large channels with thousands of messages and files can load more slowly — this is a known limitation
- Workaround: use search to jump to specific messages rather than scrolling through the channel history

**Escalation criteria:** Escalate if issue affects multiple users across different networks and there is no active incident — may be a backend performance regression.

---

### 6B: Notifications Not Working

**When to use:** Customer is not receiving Lumo notifications on desktop or mobile.

**Step 1 — Identify the notification type:**
- Desktop app notifications (banner alerts on screen)
- Mobile push notifications
- Email notifications
- In-app badge count

**Step 2 — Check Do Not Disturb:**
First check in Lumo: click profile picture > Pause notifications. If DND is enabled, notifications are suppressed. Disable it.

**Step 3 — Platform-specific troubleshooting:**

Desktop app notifications:
- Check OS notification permissions: macOS (System Settings > Notifications > Lumo) or Windows (Settings > System > Notifications)
- Lumo must be allowed to send notifications at the OS level
- Check: is the notification style set to "None" in macOS? Change to "Banners" or "Alerts"

Mobile push notifications:
- Check app notification permissions in phone settings (iOS: Settings > Lumo > Notifications; Android: Settings > Apps > Lumo > Notifications)
- Check: is the phone in Focus/Do Not Disturb mode at the OS level?
- Reinstall the Lumo app (uninstall and reinstall) if permissions seem correct but notifications still don't arrive

Email notifications:
- Check Lumo preferences: Profile > Preferences > Notifications > Email
- Set to "Immediately" if they want real-time emails, or adjust to their preference
- Check spam folder for lumo@notifications.lumo.com

**Step 4 — Channel-specific muting:**
If customer is not getting notifications from a specific channel:
- Open the channel > click the channel name > Notification preferences
- Confirm "Everything" or "@ Mentions and DMs" is selected for that channel
- Check if the channel is muted (muted channels never notify)

---

### 6C: Lumo Call/Huddle Audio or Video Issues

**When to use:** Customer can't hear others, can't be heard, or can't use video during huddles or calls.

**Step 1 — Run the built-in audio/video test:**
Direct the customer to: Profile picture > Preferences > Audio & Video > Run an audio, video and screensharing test
This test will identify whether the issue is with device permissions, hardware, or network.

**Step 2 — Check device permissions:**
The most common cause of audio/video issues is that Lumo hasn't been granted microphone or camera access:
- macOS: System Settings > Privacy & Security > Microphone (and Camera) > Enable for Lumo
- Windows: Settings > Privacy > Microphone (and Camera) > Allow apps to access
- After granting permissions: restart the Lumo app completely

**Step 3 — Check for conflicting apps:**
If another app (Zoom, Teams, Google Meet) is using the microphone or camera at the same time:
- Close the other app completely (not just minimize)
- Return to Lumo huddle

**Step 4 — VPN/firewall:**
WebRTC (used for audio/video) is sometimes blocked by corporate firewalls or VPNs:
- Try on a non-VPN connection
- If on VPN: IT may need to allow WebRTC traffic on the corporate network

**Step 5 — Noise cancellation issue:**
If noise cancellation specifically is not working or causing distortion:
- Go to Preferences > Audio & Video > Noise Cancellation > Disable
- This is a known issue on certain hardware configurations — disabling resolves audio distortion for most affected users

**Step 6 — Browser fallback:**
If issue persists on desktop app: try using Lumo in Chrome browser for calls as a temporary workaround while investigating the app issue.

---

## PLAYBOOK 7: Integration & App Issues

### 7A: Integration Stopped Working After Password Change

**When to use:** An integration (Google Drive, GitHub, Jira, etc.) was working and has stopped — typically after the user changed their password on the connected service.

**Step 1 — Explain the cause:**
> "When you change your password on [connected service], the authentication token that links it to Lumo becomes invalid. You'll need to re-authorize the connection."

**Step 2 — Re-authorization steps:**
This varies slightly by integration but the general pattern is:
- Find the integration in Lumo (the app's name in a channel, or Settings > Apps > [app name])
- Look for a "Reconnect" or "Re-authorize" button/link
- Click it and sign in to the external service again
- The integration should resume working immediately

**Step 3 — If reconnect option isn't showing:**
- Remove the integration completely (Settings > Administration > Manage Apps > [app] > Remove)
- Re-add the integration from the App Directory
- Go through the setup and authorization flow again

---

### 7B: Can't Install an App

**When to use:** Customer is trying to install an app from the App Directory but cannot.

**Step 1 — Check plan limits:**
If workspace is on Free plan:
- Maximum 10 active integrations
- If already at 10: must remove an existing one before adding a new one
- Ask: "How many integrations does your workspace currently have active?"

**Step 2 — Check admin approval setting:**
If on a paid plan:
- Ask: "When you try to add the app, does it say 'Add to Lumo' or 'Request to Install'?"
- "Request to Install" means the admin has enabled app approval — the request goes to an admin for review
- Direct the customer to ask their admin to approve the request (admins see pending approvals at Settings > Manage Apps > Pending Approvals)

**Step 3 — Check if app has been blocked:**
Admins can specifically block certain apps from being installed at all. If the app is blocked:
- Only an admin can unblock it (Settings > Manage Apps > App Policies)
- Direct customer to their IT/admin team

**Step 4 — Check if app is unavailable:**
Occasionally apps are removed from the App Directory. If the app cannot be found at all:
- Check if the integration is available through a different app name
- If the app was removed: there is no workaround — Lumo cannot restore removed third-party apps

---

### 7C: Integration Posting Duplicate Messages

**When to use:** An integration (commonly Zoom, Google Calendar, or Jira) is posting duplicate notifications to a channel.

**Step 1 — Confirm the pattern:**
Ask: "Is this every message from the integration appearing twice, or just some? Did it start after any changes — reinstalling the app, adding a new workspace, or any admin changes?"

**Step 2 — Known issue — Zoom integration:**
The Zoom integration occasionally creates duplicate meeting notifications. This is a known issue. The fix is:
1. Remove the Zoom integration from Lumo (Settings > Manage Apps > Zoom > Remove)
2. Re-add from App Directory
3. Go through the authorization flow again
This typically resolves the duplication.

**Step 3 — Multiple webhook configurations:**
For developer-configured integrations (webhooks, custom apps):
- Check whether the integration has been configured twice (two webhooks pointing to the same channel)
- Remove duplicate webhook configurations in the app settings or in the connected service

**Step 4 — Multiple admins installing the same app:**
Sometimes two admins independently install the same integration, resulting in two active instances:
- Check Settings > Manage Apps for duplicate entries of the same app
- Remove the duplicate

---

## PLAYBOOK 8: Data & Compliance

### 8A: Message History Disappeared After Downgrade

**When to use:** Customer downgraded from a paid plan to Free and can no longer see old messages.

**Step 1 — Reassure — messages are not deleted:**
This is the most important thing to communicate immediately:
> "Your messages have not been deleted. When a workspace moves to the Free plan, messages older than 90 days become hidden, but they are still there. They will reappear immediately if you upgrade to any paid plan."

**Step 2 — Explain the options:**
Option A — Upgrade to restore access:
- Any paid plan (starting with Pro at $7.25/user/month annual) immediately restores access to full message history
- If they only need to access the messages once (e.g., to export them), they can upgrade, export, and then downgrade again — they will be charged for the upgrade period

Option B — Export before any further changes:
- If they haven't already: Business+ allows full export of all message history before it is permanently deleted after 1 year on Free

**Step 3 — Warn about permanent deletion timeline:**
> "Messages and files on the Free plan are permanently deleted after they have been hidden for 1 year. To ensure nothing is permanently lost, I'd recommend acting before that deadline."

---

### 8B: How to Export Data for Compliance or Legal Discovery

**When to use:** Customer needs to export workspace data for legal discovery, regulatory audit, or internal compliance purposes.

**Step 1 — Confirm plan eligibility:**
- Business+ and Enterprise Grid: full export (all channels, private channels, DMs, files)
- Free/Pro: public channel export only — cannot export DMs or private channels

If they are on Free/Pro and need DMs/private channels:
> "A full compliance export including DMs and private channels requires Business+. Would you like to discuss upgrading?"

**Step 2 — Initiate the export:**
For Business+ or Enterprise Grid:
1. Workspace Owner or Admin goes to Settings > Administration > Workspace Settings
2. Click "Import/Export Data"
3. Select date range and channel types to include
4. Click "Start Export"
5. An email is sent to the admin's email address when the export is ready for download
6. Export time: minutes for small workspaces; up to 24 hours for large workspaces with years of history

**Step 3 — Explain the export format:**
Export is in JSON format. Key things to know:
- Messages, files, and metadata (timestamps, user IDs, edit history, delete history) are included
- Edit history is included — if a message was edited, the original text is in the export
- Deleted messages are included in the export if they were deleted by a member (admin-deleted messages may not be included depending on configuration)
- Third-party eDiscovery tools (Relativity, Logikcull, etc.) can ingest Lumo's JSON export format directly

**Step 4 — Legal hold:**
For Enterprise Grid customers who need to preserve data for litigation:
- Legal hold prevents messages in specified channels from being deleted, even if retention policies are set to auto-delete
- Configure in Settings > Administration > Legal Hold
- If they are not on Enterprise Grid and need legal hold: discuss upgrade to Enterprise Grid

---

### 8C: GDPR Data Request

**When to use:** Customer is asking about their rights under GDPR — access, erasure, portability.

**Step 1 — Clarify Lumo's role:**
> "Under GDPR, Lumo acts as a data processor — we process data on behalf of your organization, which is the data controller. This means most GDPR rights requests from individual employees should be handled by your company's data controller (typically your IT or legal team), not by Lumo directly."

**Step 2 — Right to access / data portability:**
For workspace admins:
- Business+ and Enterprise Grid: use the Export tool (see Playbook 8B)
- Free/Pro: export public channel data

For individual employees asking for their personal data:
- Direct to their company's GDPR officer or IT team — they control the data
- Lumo's DPA is available at lumo.com/legal/dpa

**Step 3 — Right to erasure:**
Individual message deletion:
- Members can delete their own messages within 24 hours
- Admins can delete any message at any time
- Mass deletion of a departed member's messages: submit formal request to privacy@lumo.com
- Processing time: up to 30 days
- Note this limitation honestly: individual messages in channels cannot be surgically deleted from exports retrospectively — the export already captured those messages

**Escalation criteria:** If the customer mentions GDPR enforcement action or supervisory authority involvement, escalate to senior agent immediately and loop in Lumo's legal/privacy team.

---

## PLAYBOOK 9: Workspace Management

### 9A: Ownership Transfer

**When to use:** The workspace ownership needs to be transferred — either the current owner is transferring it voluntarily, or the owner has left the company and the new admin needs access.

**Case A — Voluntary transfer (current owner available):**
Path: Settings > Administration > Workspace Settings > Transfer Ownership
- Current owner selects new owner from member list
- New owner receives email confirmation request
- Both parties confirm
- Transfer is immediate

**Case B — Owner has left the company:**
This requires Lumo Support involvement. Required documentation:
1. Proof of company domain ownership: a DNS TXT record verification OR a business registration document showing the requester's authority over the company domain
2. Email of the new owner (must have a matching company domain email)
3. Workspace URL

Process:
1. Customer submits request with documentation to support@lumo.com
2. Support team verifies documentation (1–3 business days)
3. If verified: Support transfers ownership to the new owner
4. New owner receives email confirmation and can access the workspace

Tell the customer: "Once you can provide proof that you are authorized to represent this organization (such as a verification of the company domain), we can facilitate the ownership transfer. This typically takes 1–3 business days once we have the documentation."

**Escalation criteria:** All Case B ownership transfers must be reviewed by a senior agent — do not process without senior approval.

---

### 9B: Workspace URL Change

**When to use:** Customer wants to change their workspace URL (lumo.com/workspace-name).

**Step 1 — Warn about the consequences — this is important:**
Before proceeding, make sure the customer fully understands:
> "Changing your workspace URL will immediately break all existing Lumo links using the old URL — there is no automatic redirect. This includes desktop app bookmarks, any links shared in documents or emails, and browser shortcuts your team members have saved. All team members will need to update to the new URL."

**Step 2 — Confirm they want to proceed:**
Explicitly ask: "Are you aware that the old URL will stop working immediately? Would you like to proceed?"

**Step 3 — Walk through the change:**
Settings > Administration > Workspace Settings > Workspace URL > Edit
- The new URL is available immediately if not already taken
- URL can only be changed once every 90 days — after changing, no further URL changes for 90 days

**Step 4 — Post-change steps to communicate:**
After changing:
- Share the new URL with all workspace members
- Update the Lumo app on desktop (may need to re-add the workspace with new URL)
- Update any integrations that use the workspace URL in their webhook configuration
- Update any external links or documentation that contained the old URL

---

## PLAYBOOK 10: Escalation Reference

### When to Escalate Immediately (Before Attempting Resolution)

- Customer mentions legal action, lawsuit, or regulatory complaint
- Potential data breach or unauthorized account access
- Any P1 issue (complete workspace outage) not resolved within 30 minutes
- Enterprise Grid customer with any priority issue
- Identity cannot be verified but customer is claiming to be workspace owner
- Refund request outside 30-day window (requires senior approval before any commitment)

### When to Escalate After Initial Troubleshooting

- Issue cannot be reproduced or root cause not identified after following the playbook
- Confirmed bug requiring engineering investigation
- Billing discrepancy that appears to be a system error (incorrect charge amounts)
- GDPR or legal hold inquiry
- Ownership transfer where owner has left the company (Case B above)

### What to Include When Escalating

Always provide the following when escalating:
1. Customer name and contact information
2. Workspace name and URL
3. Plan tier
4. Steps already taken and results
5. Any error messages verbatim
6. Priority assessment (P1/P2/P3/P4/P5)
7. Whether you have made any commitments to the customer (e.g., "I told them a refund was likely")

### Tone During Escalation

When telling a customer you're escalating:
> "I want to make sure this gets the right attention. I'm escalating your case to our senior support team who can [specific action]. You'll hear back within [realistic timeframe]. I've made sure they have all the details from our conversation so you won't need to repeat yourself."

Do not say: "I can't help you with this." Instead say: "I'm getting you to someone who can resolve this completely."
