# LUMO — SUPPORT POLICIES
**Version 1.0 | Internal Support Knowledge Base**

---

## Overview

This document defines Lumo's official support policies. All customer-facing agents must follow these policies exactly. Where a policy has exceptions, the exceptions are explicitly listed. If a situation is not covered by this document, escalate to a senior agent before making commitments to the customer.

These policies apply to all Lumo plans unless a specific plan is called out.

---

## 1. Billing & Refund Policies

### 1.1 Standard Refund Window

Lumo offers a **30-day money-back guarantee** on the first payment only for new Pro and Business+ subscriptions.

**Eligibility requirements — ALL must be true:**
- The refund request is made within 30 calendar days of the first charge
- It is the first time this workspace has been on a paid plan (not a re-upgrade after a previous cancellation)
- The plan being refunded is Pro or Business+ (Enterprise Grid refunds are handled by the account manager and are non-standard)
- The workspace owner or billing admin is making the request

**What is NOT eligible for the standard refund:**
- Renewal charges (second month, second year, etc.) — the 30-day window applies to first payment only
- Mid-cycle seat additions (prorated charges when new members are added)
- Annual plan renewal charges, even if the customer forgot they were on annual billing
- Downgrades that reduce the bill going forward — these are not refunded, they take effect at next renewal
- Free plan — there is nothing to refund
- Plan upgrades where the customer has been actively using the new features for more than 30 days

**How to process a standard refund:**
1. Verify eligibility against all criteria above
2. Confirm the workspace name, billing email, and charge amount
3. Process refund via the billing system — note: refunds are issued to the original payment method only
4. Inform customer: refund takes **5–7 business days** to appear on their statement
5. Log the refund in the ticket with amount, date, and authorization

---

### 1.2 Courtesy Refunds (Outside Standard Window)

In limited cases, a courtesy refund may be issued outside the 30-day window. These require senior agent approval and must meet at least one of the following criteria:

**Accidental workspace creation:**
- Customer created a new workspace by mistake (e.g., signed up instead of joining an existing workspace)
- Must be requested within **14 days** of the accidental charge
- Workspace must be unused or minimally used (fewer than 5 messages posted total)
- One courtesy refund per customer account, lifetime

**Documented billing error:**
- Lumo charged an incorrect amount due to a system error
- Lumo charged after a confirmed cancellation was in effect
- Lumo charged for seats that were never added (system error, not admin action)
- No time limit — billing errors are always corrected

**Unauthorized account creation:**
- An employee created a paid workspace using company card without authorization
- Company must provide evidence (e.g., email from IT or HR confirming the employee acted without authorization)
- Must be requested within **30 days** of discovery, not charge date
- Workspace is deactivated as part of the resolution

**What is NOT a courtesy refund scenario:**
- Customer forgot they were on annual billing
- Customer didn't use Lumo much and wants money back
- Customer found a cheaper alternative
- Admin added seats and the customer didn't notice
- Customer disagrees with the price increase

---

### 1.3 Duplicate Charges

If a customer is charged twice for the same billing period, this is always corrected with no questions asked.

**Process:**
1. Ask customer to confirm: workspace name, the two charge dates and amounts, last 4 digits of card
2. Verify in billing system that two charges exist for the same period
3. Refund the duplicate charge immediately — no approval needed
4. Refund timeline: **5–7 business days**
5. If the cause was a system error, flag for engineering

---

### 1.4 Charged After Cancellation

If a customer was charged after their cancellation was confirmed and effective:

1. Verify that cancellation was completed before the charge date (check cancellation timestamp in system)
2. If confirmed: issue full refund for the post-cancellation charge — no approval needed
3. If cancellation was submitted but workspace shows no cancellation recorded: treat as courtesy refund with senior approval
4. Inform customer of refund timeline (5–7 business days)

**Important distinction:** If the customer cancelled and the cancellation is effective at the next renewal date (standard behavior), a charge on that renewal date is correct — the workspace was still active until that date. This is not a refund scenario. Explain the cancellation effective date to the customer.

---

### 1.5 Annual vs Monthly Billing Refunds

- **Monthly billing:** standard 30-day window applies to first month only
- **Annual billing:** standard 30-day window applies to first annual charge only
- After the 30-day window on an annual plan: no refund for unused time remaining in the year
- Customer can downgrade or cancel, effective at end of the annual period — they keep access for the remainder of the year they paid for
- Exception: documented billing error (see 1.2) — always corrected regardless of billing type

---

### 1.6 Refund Method

Refunds are always issued to the original payment method used for the charge. Lumo cannot:
- Issue refunds to a different card
- Issue refunds as bank transfers or checks
- Issue refunds as Lumo credits (unless the customer specifically requests account credit instead of a card refund, which is permitted)

If the original card is no longer active: customer's bank typically routes the refund to the replacement card associated with the same account number. Advise customer to contact their bank if refund doesn't appear after 10 business days.

---

## 2. Plan Change Policies

### 2.1 Upgrades

**Timing:** Upgrades take effect immediately upon confirmation.

**Billing:** When upgrading mid-cycle, the customer is charged a prorated amount for the remainder of the current billing period at the new (higher) rate. On the next billing date, they are charged the full new plan rate.

**Example:** Customer on Pro ($7.25/user/month annual) upgrades to Business+ ($15/user/month annual) on the 15th of a 30-day month. They are charged for 15/30 days × ($15 - $7.25) = $3.875/user as a proration charge immediately. Their next full charge on their regular date is at the Business+ rate.

**Seat additions during upgrade:** If a customer also adds seats at upgrade time, both the plan upgrade proration and the new seat proration are charged simultaneously.

**Who can initiate an upgrade:** Workspace Owner and Admins with billing access.

---

### 2.2 Downgrades

**Timing:** Downgrades do NOT take effect immediately. They take effect at the **next billing cycle renewal date**.

**What this means for customers:** After requesting a downgrade, the customer continues on their current (higher) plan and retains all current features until their renewal date. They are NOT charged the lower rate until the next cycle begins.

**No partial refunds for downgrades:** The customer has paid for the current period. The downgrade is a forward-looking change only.

**Feature loss on downgrade — important to communicate:**

| Downgrade Path | Features Lost |
|---|---|
| Business+ → Pro | SAML SSO, SCIM provisioning, compliance exports, advanced AI, 99.99% SLA |
| Business+ → Free | All of the above + message history (90-day limit applies, old messages hidden) |
| Pro → Free | Unlimited message history (90-day limit applies), unlimited integrations limited to 10, group calls, unlimited workflows |
| Enterprise → Business+ | Multi-workspace, data residency, HIPAA BAA, eDiscovery, dedicated support |

**Always warn customers about message history on downgrade to Free:** Messages older than 90 days will become hidden (not deleted) immediately upon the downgrade taking effect. They are NOT lost — upgrading again restores full access immediately.

---

### 2.3 Seat Count Changes

**Adding seats:**
- Takes effect immediately
- Prorated charge for the remainder of the current billing period
- Minimum on Pro and Business+: 3 seats (even if workspace has fewer active members)

**Removing seats:**
- Takes effect at the next billing cycle renewal date
- No credit or refund for seats removed mid-cycle
- If removing seats would bring count below 3: minimum 3 seats still applies on Pro and Business+
- Deactivating a member in Lumo does NOT automatically reduce the seat count in billing — admin must also remove the seat in billing settings

**Important distinction:** Deactivating a member (removing access) and reducing the paid seat count are two separate actions. Many customers don't realize this. Clarify: deactivating a member stops them from logging in but the seat remains billable until removed in billing settings.

---

### 2.4 Cancellations

**How to cancel:** Workspace Owner > Settings > Administration > Billing > Cancel Plan.

**What happens immediately:** Nothing changes immediately. The workspace remains on the current paid plan until the end of the current billing period.

**What happens at the end of the paid period:**
- Workspace reverts to Free plan behavior (90-day message history limit, 10-app integration limit, etc.)
- No data is deleted immediately
- Workspace remains accessible on Free plan

**What happens to data after cancellation:**
- Messages and files are retained for **90 days** after the workspace reverts to Free status
- After 90 days on Free: messages over 90 days old begin to be permanently deleted on a rolling basis
- After 1 year on Free: messages and files over 1 year old are permanently deleted
- To preserve data: export before cancelling (Business+ required for full export; Pro/Free only exports public channels)

**Reactivation after cancellation:**
- Customer can re-upgrade at any time
- Data is restored if within the retention window
- If reactivating after data deletion: only data from after reactivation date is available

---

### 2.5 Trial Period Policies

**Standard trial:** New workspaces on Pro get a 90-day free trial of Pro features before being asked to add a payment method.

**Trial extensions:**
- Available for up to 30 additional days
- Eligibility: workspace must have 10+ active members AND the team must not have received a previous extension
- Must be requested before trial expiry — cannot be applied retroactively
- Approved at agent discretion for Pro-tier trials; Business+ trial extensions require senior agent approval
- To apply: note the extension in the ticket and apply in the billing system

**Trial to paid conversion:**
- At trial end, workspace is prompted to enter payment details
- If no payment is entered, workspace reverts to Free plan — not cancelled, just Free
- Trial usage counts as actual usage — if the customer triggered the 30-day refund window assumption by entering a card mid-trial, clarify with senior agent

---

## 3. Payment & Invoice Policies

### 3.1 Accepted Payment Methods

- Credit and debit cards: Visa, Mastercard, American Express, Discover
- ACH bank transfer: available for annual plans of 25+ seats (US accounts only)
- Invoice/Purchase Order: available for Enterprise Grid customers and annual plans of 50+ seats on Business+
- Lumo does NOT accept PayPal, cryptocurrency, or prepaid cards

### 3.2 Failed Payments

When a payment fails:
1. Lumo sends an email notification to the billing email address on file
2. Lumo retries the charge automatically: after 3 days, then after 5 days, then after 7 days
3. If all retries fail, the workspace is downgraded to Free plan
4. The workspace is NOT deleted — it reverts to Free

**Customer resolution steps:**
1. Update payment method in Settings > Administration > Billing
2. Manually trigger a charge retry, OR wait for the next automatic retry
3. If workspace has already been downgraded: updating payment method and paying the outstanding balance restores the paid plan immediately

**Grace period:** There is a 7-day grace period from first failed payment notification before the workspace is downgraded. During this period, full plan features remain active.

---

### 3.3 Invoice & Receipt Requests

- Invoices are automatically emailed to the billing email after each successful charge
- Workspace Owners and Admins can download all past invoices from Settings > Administration > Billing > Invoice History
- Invoice format: PDF, includes workspace name, workspace ID, billing period, number of seats, plan name, unit price, tax line items, total charged

**If customer can't access billing settings:**
- Support can resend any past invoice to the billing email on file
- Support cannot send invoices to a different email address without owner verification
- For tax purposes requiring a different billing name or address on the invoice: admin must update billing details first, then future invoices reflect the new information — retroactive invoice changes are not possible

**Invoice line items that commonly cause confusion:**

| Line Item | Explanation |
|---|---|
| Base plan charge | Regular per-seat monthly or annual charge |
| Seat addition proration | Mid-cycle charge when seats were added |
| Plan upgrade proration | Mid-cycle charge when plan was upgraded |
| Lumo Connect | Charges for Connect features in certain configurations |
| Tax | State/local sales tax applied based on billing address |
| Credit applied | A credit from a previous overpayment or promotional credit being used |

---

### 3.4 Tax Policies

- Lumo collects sales tax in US states where required by law
- Tax rate is determined by the billing address on the account
- Customers with tax-exempt status (nonprofits, government entities, educational institutions) can submit their exemption certificate to billing@lumo.com — tax will be removed from future invoices
- Lumo cannot retroactively remove tax from past invoices once paid
- International customers: Lumo does not collect VAT — international customers are responsible for self-assessing VAT in their jurisdiction

---

## 4. Security & Authentication Policies

### 4.1 Two-Factor Authentication (2FA)

**Available 2FA methods:**
- Authenticator app (Google Authenticator, Authy, 1Password, etc.) — recommended
- SMS text message (less secure, but available)

**2FA availability by plan:**
- All plans: members can enable 2FA individually
- Business+ and Enterprise Grid: admins can enforce 2FA for all workspace members

**If a member is locked out due to 2FA:**

Scenario A — Member lost their authenticator device:
- The workspace Admin or Owner can temporarily disable 2FA for that specific member
- Path: Admin > Settings > Members > [member name] > Security > Disable 2FA
- Member can then log in and set up 2FA on new device
- Agent does NOT need to be involved unless the Admin is also locked out

Scenario B — Admin or Owner is locked out due to 2FA:
- Contact Lumo Support
- Support will verify identity via: billing email on file + last 4 digits of card on file + workspace URL
- Upon verification, Support can disable 2FA for the account to restore access
- Resolution time: within 1 business day for Pro/Business+; priority same-day for Enterprise Grid

Scenario C — 2FA enforced org-wide, member can't set it up:
- Advise admin to temporarily turn off org-wide 2FA enforcement
- Member sets up 2FA on their device
- Admin re-enables enforcement
- If admin can't do this, Support can assist with Business+ or Enterprise Grid

**SMS 2FA not receiving codes:**
- Check: is the phone number correct and formatted with country code?
- Check: is the phone capable of receiving SMS from US short codes?
- Check: carrier may be blocking automated SMS — advise switching to authenticator app
- Support cannot resend SMS verification codes — this is handled by the platform automatically

---

### 4.2 Single Sign-On (SSO)

**Google Workspace SSO:**
- Available on ALL plans including Free
- Sign in with Google — uses the member's Google account associated with the workspace
- No Lumo-specific password needed when Google SSO is enabled
- Issue: if Google Workspace admin restricts third-party app access, members may be blocked from Lumo — this is a Google admin issue, not a Lumo issue

**SAML-based SSO (Okta, Microsoft Entra ID, OneLogin, etc.):**
- Requires Business+ or Enterprise Grid
- Configured in Settings > Authentication > SAML Configuration
- Requires: IdP metadata XML or SSO URL and certificate from the identity provider

**Common SAML SSO issues and resolutions:**

| Issue | Likely Cause | Resolution |
|---|---|---|
| "SAML response is invalid" | Expired certificate in IdP | Renew certificate in IdP and re-upload metadata to Lumo |
| "User not found" after SSO | Email in IdP doesn't match email in Lumo | Ensure IdP attribute mapping sends the correct email field |
| Login loop (redirects infinitely) | ACS URL mismatch | Verify ACS URL in IdP matches exactly what Lumo provides |
| SSO works for some users but not others | Users not assigned to Lumo app in IdP | Assign all Lumo users to the Lumo application in IdP |
| Locked out after enabling SSO | Admin used non-SSO account, now can't bypass | Contact Lumo Support — Support can disable SSO via backend for Business+/Enterprise |

**Emergency SSO bypass:**
- If SAML SSO is configured incorrectly and locks all users out: Support can disable SSO for the workspace upon verification of Owner identity
- Verification: billing email + workspace URL + last 4 of card on file
- This is only for Business+ and Enterprise Grid — Pro/Free do not have SAML SSO

---

### 4.3 Password Policies

**Password reset flow:**
1. Customer goes to lumo.com/signin and clicks "Forgot password"
2. Enters their email address
3. Receives a magic link via email (valid for 30 minutes)
4. Clicks link and sets new password

**If password reset email is not received:**
- Check spam, junk, and promotions folders
- Check if the email address entered is correct
- Check if the organization enforces SSO — if so, password reset goes through the IdP (Okta, Google, etc.), not Lumo
- Corporate email security filters sometimes strip magic links — advise IT to whitelist emails from no-reply@lumo.com
- Resend the reset email (up to 3 times — after that, wait 1 hour before another can be sent)

**Lumo Support cannot:**
- Reset a password directly on behalf of a user
- Provide a user's current password
- Bypass the email-based reset flow (except for 2FA lockout — see 4.1)

---

### 4.4 Account Verification & Identity

When Lumo Support needs to verify a customer's identity before making account changes (e.g., ownership transfer, 2FA reset for admin, SSO disable):

**Standard verification (Pro/Business+):**
- Billing email address on file
- Last 4 digits of the payment card on file
- Workspace URL (lumo.com/workspace-name)

**Enhanced verification (Enterprise Grid):**
- All of the above, plus
- Account manager confirmation OR
- Written authorization from the account's primary contact on file

Support should never make account changes without completing verification. If a customer cannot verify, do not proceed — ask them to contact their company's IT admin or billing contact who can verify.

---

## 5. Data & Compliance Policies

### 5.1 Data Access by Lumo Support

**Lumo Support cannot read workspace messages or files.** This is a hard technical and policy limit. Support agents do not have access to the content of any workspace — messages, files, canvases, or DMs.

Support CAN see:
- Workspace metadata (name, plan, creation date, member count, billing status)
- Billing history (charges, refunds, invoices)
- Admin activity logs at a high level (actions taken, not content)
- Technical logs related to errors (without message content)

This means if a customer asks "can you see if a message was sent?" — the answer is no. Support can confirm whether a member was active or connected but cannot verify message content.

---

### 5.2 GDPR & Data Subject Requests

Lumo acts as a data processor. The customer (workspace owner/admin) is the data controller. This means:

**Right to access / Right to export:**
- Business+ and Enterprise Grid: admins can export all data via the Export tool
- Free/Pro: admins can export public channel data only
- For individual member data export requests: the workspace admin handles this under their own GDPR obligations — Support directs them to the export tool
- Lumo's DPA (Data Processing Agreement) is available at lumo.com/legal/dpa

**Right to erasure (right to be forgotten):**
- Deactivating a member removes their access — this is NOT the same as deletion
- For actual deletion of a member's data: workspace admin can submit a formal erasure request via privacy@lumo.com
- Processing time: up to 30 days
- Note: messages in channels cannot be deleted individually at scale — this is a platform limitation
- Enterprise Grid: custom data retention policies can automate deletion after a set period

**Data Portability:**
- Business+/Enterprise: full export in JSON format covers data portability requirements
- Support directs customers to the export tool and Lumo's DPA documentation

---

### 5.3 Compliance Export Policy (Business+ and Enterprise Grid)

The compliance export feature allows workspace admins to export all messages, files, and DMs for legal discovery, auditing, or regulatory compliance.

**Who can initiate a compliance export:** Only the Workspace Owner or Primary Admin.

**What is included:**
- All public channel messages
- All private channel messages
- All direct messages (1:1 and group DMs)
- All files shared in any of the above
- Edit and delete history (if a message was edited or deleted, the original is included)

**What is NOT included:**
- Drafts (unsent messages)
- Lumo Connect messages from external organizations (only your side of the conversation)
- Scheduled messages that haven't been sent yet

**Export format:** JSON. Multiple files split by channel and date range. Third-party tools (eDiscovery platforms) can ingest this format directly.

**Time to complete:** Depends on workspace size. Small workspaces: minutes. Large workspaces (years of history, thousands of members): can take up to 24 hours. Customer receives an email when export is ready to download.

**Agent note:** If a customer on Free or Pro requests a compliance export (full DMs and private channels), inform them this requires Business+. Do not attempt to process the export for Free/Pro — the feature is not enabled for their plan.

---

### 5.4 HIPAA & BAA Policy

HIPAA compliance and a Business Associate Agreement (BAA) are available **only on Enterprise Grid**.

If a customer on Pro or Business+ asks about HIPAA:
- Explain that HIPAA compliance via BAA requires Enterprise Grid
- Do NOT tell them their current plan is HIPAA compliant — it is not
- Connect them with the sales team for an Enterprise Grid evaluation
- Do not speculate about whether their specific use case "probably" is fine without a BAA — always recommend proper legal review

---

### 5.5 Data Retention Policy Details

**Default retention (all paid plans):** Messages and files are kept indefinitely.

**Custom retention (Business+ and Enterprise Grid):**
- Admins can set retention periods per workspace or per channel
- Example: set #general to retain messages for 1 year, then auto-delete
- Minimum retention period: 1 day (not recommended except for highly sensitive channels)
- Retention policy applies prospectively from when it is set — it does NOT retroactively delete older messages that were sent before the policy was configured

**Retention on Free plan:**
- Messages older than 90 days: hidden from search but not deleted
- Messages older than 1 year: permanently deleted on a rolling basis
- Files: subject to 5 GB total storage limit — oldest files may be removed when limit is reached

**After workspace cancellation / reversion to Free:**
- The workspace retains access to all messages for 90 days after reverting to Free (grace period for export)
- After 90 days on Free: 90-day rolling visibility applies, older messages begin being hidden
- If no upgrade occurs: messages over 1 year old are permanently deleted

---

## 6. Integrations & Third-Party App Policies

### 6.1 App Installation & Approval

**Free plan:** Maximum 10 active app integrations at any time. Installing an 11th requires removing an existing one.

**Pro and above:** Unlimited app integrations.

**App approval flow (Business+ admin control):**
When an admin enables app approval requirements:
- Members see "Request to install [App Name]" instead of "Add to Lumo"
- Request is sent to admins for review
- Admin approves or denies in Settings > Administration > Manage Apps > Pending Approvals
- If denied: the member is notified and cannot install the app
- If approved: the app is installed automatically

**Agent note:** If a member says they can't install an app, the most common causes are: (1) workspace is on Free with 10 apps already, (2) admin has enabled app approval and hasn't approved the request yet, (3) the app itself has been blocked by the admin.

---

### 6.2 OAuth & Authorization Tokens

Many integrations use OAuth to connect a user's external account (Google Drive, GitHub, etc.) to Lumo. These tokens can expire or become invalid if:
- The user changes their password on the external service
- The user revokes Lumo's access in the external service's connected apps settings
- The external service requires periodic re-authorization
- The admin removes the app and re-adds it

**Resolution:** The user must reconnect/re-authorize the integration. In most apps this is done by clicking the app in Lumo and following the "Reconnect" or "Re-authorize" prompt.

**Support cannot:** Directly refresh or reset OAuth tokens. The user must go through the re-authorization flow themselves.

---

### 6.3 Lumo Connect Policies

**Eligibility:** Both the sending and receiving organization must be on a paid Lumo plan.

**Channel Connect (shared channels):**
- Shared channels appear in both workspaces' channel lists
- Messages are searchable in both workspaces
- If the receiving workspace is on Free: the invitation cannot be accepted until they upgrade to a paid plan
- Removing an organization from a Connect channel: the channel disappears from their workspace, but messages they sent remain in your workspace's history

**DM Connect:**
- Can send DMs to members of other Lumo organizations
- Requires the other person's email address associated with their Lumo account

**Pricing for Connect:**
- Standard Connect features: included in Pro and above
- Some high-volume Connect configurations (typically relevant to Enterprise Grid) may have additional usage charges — these appear on invoices as "Lumo Connect" line items
- If a customer sees an unexpected Connect charge: verify whether they have active Connect channels or DM Connect in use

---

## 7. Workspace Management Policies

### 7.1 Workspace Deletion

Only the Workspace Owner can delete a workspace. This action is **permanent and irreversible**.

**Before deleting:** All data (messages, files, canvases) is permanently deleted. The workspace URL becomes available for re-use by anyone. There is no recovery after deletion.

**If a customer wants to delete their workspace:**
- Confirm they understand this is permanent
- Confirm they have exported any data they need (requires Business+ for full export)
- Direct them to: Settings > Administration > Workspace Settings > Delete Workspace
- Support cannot delete a workspace on behalf of a customer — only the Owner can

**If the Owner has left the company:**
- Support can facilitate an ownership transfer to a current employee
- Requires: proof of company domain ownership (DNS TXT record or business registration)
- Once ownership is transferred, the new Owner can delete the workspace
- Processing time: 1–3 business days

---

### 7.2 Workspace Name and URL Changes

**Workspace display name:** Owner or Admin can change at any time in Settings > Administration > Workspace Settings.

**Workspace URL (lumo.com/workspace-url):**
- Can only be changed once every 90 days
- After changing: the old URL stops working immediately — there is no redirect
- All existing Lumo app links (desktop app bookmarks, browser shortcuts) using the old URL will break
- Members using the desktop app may need to re-add the workspace using the new URL
- **Agent note:** Warn customers strongly about broken links before changing workspace URL. It cannot be undone for 90 days.

---

### 7.3 Workspace Merges

Lumo does not provide a native workspace merge tool. If an organization has two workspaces they want to combine:
- This must be done manually (invite members from one workspace to the other, export/import content as needed)
- Support can advise on best practices but cannot perform the merge
- For Enterprise Grid customers: the account manager can discuss multi-workspace migration options

---

## 8. Guest User Policies

### 8.1 Single-Channel Guests

- Can participate in exactly one channel
- Cannot see any other channels, DMs, or workspace members they aren't in contact with
- Free on all paid plans (Pro, Business+, Enterprise Grid)
- Can be invited by any Admin
- Cannot be elevated to Member or Admin role
- Do not count toward minimum seat count

### 8.2 Multi-Channel Guests

- Can participate in multiple channels (set at invitation time)
- Count as full paid seats — same pricing as regular members
- Have the same billing implications as adding a regular member
- Cannot be elevated to Admin role

### 8.3 Guest Expiry

- Guests can be given an expiry date at invitation time
- After the expiry date, they are automatically deactivated
- If no expiry is set, they remain active until manually deactivated
- Reminder emails about guest expiry are sent to the admin who invited them

---

## 9. Support SLA & Response Time Policies

### 9.1 Response Time Commitments

| Plan | Channel | Target Response Time |
|---|---|---|
| Free | Help Center / Forums only | No SLA — self-service only |
| Pro | Email / Chat | 1–2 business days (best effort) |
| Business+ | Email / Chat (priority queue) | Same business day for critical issues; 1 business day standard |
| Enterprise Grid | Dedicated account manager | 4 hours for critical (P1/P2); 1 business day for standard |

**Business hours:** Monday–Friday, 9am–6pm in all major time zones (US, UK, EU, APAC). Enterprise Grid critical support is 24/7.

**Note on chat support:** Chat is available during business hours for Pro and above. Outside business hours, chat goes to an AI assistant that can resolve common issues and create a ticket for human follow-up.

---

### 9.2 Issue Priority Levels

| Priority | Definition | Example |
|---|---|---|
| P1 — Critical | Complete service outage; unable to send or receive messages | Workspace inaccessible for all members |
| P2 — High | Major feature broken; significant portion of users affected | All notifications stopped; SSO broken for org |
| P3 — Medium | Single feature degraded; workaround available | Search returning incomplete results; one integration broken |
| P4 — Low | Minor inconvenience; cosmetic or edge-case issue | Notification badge count incorrect; minor UI bug |
| P5 — Inquiry | Billing question, policy question, how-to | "How do I export messages?" "Can I get a refund?" |

Agents should classify incoming issues by priority and escalate P1/P2 to senior agent immediately. P5 inquiries should be resolved in the first response.

---

### 9.3 Escalation Policy

**When to escalate to senior agent:**
- Customer is requesting a refund outside the standard 30-day window
- Customer is threatening legal action or regulatory complaint
- Issue involves data loss (messages or files deleted unexpectedly)
- Issue involves a potential security incident (unauthorized access, suspected account takeover)
- P1 or P2 issue that has not been resolved within 2 hours of first contact
- Enterprise Grid customer with any P1/P2 issue

**When to escalate to engineering:**
- Confirmed bug that cannot be resolved by configuration change or workaround
- Billing system error causing incorrect charges
- Data integrity issue (messages appearing corrupted, edits not saving)

**Customer communication during escalation:**
- Always tell the customer you are escalating and give a realistic timeframe
- Do not promise specific resolution times unless you have confirmation from senior agent or engineering
- Do not tell a customer their issue is "unique" or "never seen before" — this undermines confidence

---

## 10. Promotional & Discount Policies

### 10.1 Nonprofit & Education Discounts

Lumo offers **85% off** Pro and Business+ plans for qualifying nonprofits and educational institutions.

**Eligibility:**
- Registered nonprofit organizations (501(c)(3) in the US, equivalent internationally)
- K-12 schools, colleges, and universities
- Not available to government agencies or healthcare organizations (they must use standard pricing)

**How to apply:**
- Submit application at lumo.com/nonprofits
- Include: organization name, EIN or registration number, primary contact email
- Review time: up to 5 business days
- Discount is applied to the workspace, not retroactively to past charges

**Agent note:** Do not apply nonprofit/education discounts manually without an approved application in the system. If a customer asks whether they qualify, direct them to the application page — do not make eligibility determinations yourself.

---

### 10.2 Promotional Codes

**Valid promo codes:**
- Applied at checkout when upgrading or at workspace creation
- Typically percentage discounts (e.g., 20% off first 3 months) or flat credits
- Each code has: an expiry date, a maximum number of uses, eligible plans, and eligible customer types (new vs existing)

**If a promo code is not working:**
- Ask customer for the exact code they are entering
- Check: is the code expired?
- Check: is the customer on an eligible plan? (some codes are for new workspaces only)
- Check: has the code already been used on this account? (one use per account)
- Check: is the customer billing monthly when the code requires annual?
- If the code is valid and still not working: escalate — do not manually apply discounts without senior approval

**Agent note:** Never promise a discount or apply a promotional credit without proper authorization. If a customer claims they received a promotion via email, ask them to forward the email — verify before applying.

---

### 10.3 Referral Credits

Lumo's referral program (if active) gives credits when an existing customer refers a new paying workspace.

- Credits are applied automatically to the referrer's billing account
- Credits reduce future charges — they are not refundable as cash
- Credits appear on the invoice as "Referral credit applied"
- If a referral credit is missing: verify in billing system that the referred workspace completed a qualifying upgrade, and that the referral link was properly attributed

---

## 11. Known Issues & Platform Limitations

Performance and speed issues are among the most common technical support requests. Common symptoms include the Lumo desktop app loading slowly or freezing when switching channels, search timing out or returning no results, huddle audio lagging or dropping, file uploads getting stuck at 0%, and the app crashing on startup after OS updates. These performance issues are typically caused by VPN interference blocking WebRTC traffic, outdated Lumo app versions, browser cache buildup, hardware conflicts when other apps are using the microphone or camera simultaneously, or corporate firewall rules blocking Lumo domains. For persistent performance issues, always check status.lumo.com for active incidents before troubleshooting.

The following are documented platform limitations that are by design — these are not bugs:

| Limitation | Details |
|---|---|
| Cannot delete your own messages after 24 hours | Members can edit messages anytime, but message deletion by the sender is limited to 24 hours. Admins can delete any message. |
| Cannot leave a 1:1 DM | DMs persist in the sidebar permanently. Only option is to mute. |
| Minimum 3 seats on Pro/Business+ | Even a 2-person workspace pays for 3 seats. |
| Free plan 10-app limit is strict | The 11th app cannot be added without removing an existing one — there is no grace period. |
| Workspace URL change breaks all existing links | Old URL gives a 404 immediately — no redirect. |
| Cannot merge two workspaces | No native tool exists. Must be done manually. |
| Search indexing lag | New messages may take up to 48 hours to appear in search results. |
| Cannot permanently delete member data en masse | Only deactivation is self-serve. Full data deletion requires a formal privacy request. |
| Compliance export does not include Lumo Connect messages from external orgs | Only your side of the conversation is exported. |
| SCIM only deactivates users, does not delete them | SCIM provisioning removes access but messages and history remain until manual deletion. |
| Conditional branching in workflows requires Business+ | Basic Workflow Builder is available on all paid plans, but advanced conditional branching requires Business+. |
| AI features have inherent inaccuracy | Summaries and AI answers are best-effort. Lumo does not guarantee accuracy of AI-generated content. |
