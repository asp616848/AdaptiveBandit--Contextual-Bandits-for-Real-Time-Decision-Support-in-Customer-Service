# LUMO — COMPANY & PRODUCT OVERVIEW
**Version 1.0 | Internal Support Knowledge Base**

---

## 1. Company Background

**Company name:** Lumo  
**Product:** Cloud-based team messaging and collaboration platform  
**Tagline:** Where your team's work lives  
**Founded:** 2015  
**Headquarters:** San Francisco, CA  
**Employees:** ~2,400 globally  
**Customer base:** 600,000+ organizations worldwide, from 2-person startups to Fortune 500 enterprises  
**Parent company:** Lumo operates as an independent subsidiary of Apex Cloud (acquired 2022)

Lumo is a team communication platform built around organized channels, direct messaging, file sharing, and deep integrations with the tools teams already use. It replaced internal email for hundreds of thousands of organizations and has become the central hub where decisions get made, projects move forward, and institutional knowledge lives.

Lumo competes primarily with Microsoft Teams (bundled with Office 365), Google Chat (bundled with Google Workspace), and Discord (popular with smaller technical teams).

---

## 2. Plans & Pricing (Current as of 2026)

Lumo offers four plans. All pricing is per active member per month. Guests are counted differently (see Section 4 for guest policy).

### 2.1 Free Plan — $0

**Best for:** Small teams, individuals, testing Lumo before committing.

**Key limitations:**
- Message history: only the last 90 days are searchable. Messages older than 90 days become hidden but are NOT deleted — they reappear instantly if you upgrade. Messages over 1 year old are permanently deleted on the Free plan.
- App integrations: maximum 10 active third-party app integrations at any time
- Calls: 1:1 voice and video only (no group calls)
- File storage: 5 GB shared across the entire workspace
- Workflows: Workflow Builder available but limited to 10 active automations
- AI features: not available
- Support: Help Center and community forums only — no direct human support

**Who typically hits Free plan limits:** Small teams that grow past 10 people, teams that need to reference conversations older than 90 days, teams with more than 10 app integrations.

---

### 2.2 Pro Plan — $8.75/user/month (billed monthly) or $7.25/user/month (billed annually)

**Best for:** Growing teams that need full message history and real integrations.

**Minimum charge:** 3 users ($21.75/month on monthly billing). Even if you have 2 members, you pay for 3.

**Everything in Free, plus:**
- Message history: unlimited, fully searchable forever
- App integrations: unlimited
- Calls: group voice and video calls up to 50 participants, screen sharing
- File storage: 10 GB per member
- Workflow Builder: unlimited active automations
- AI features: basic — conversation summaries, thread summaries, Lumo AI answers (uses workspace data to answer questions)
- Huddle notes: AI-generated notes and action items from voice/video calls
- Guest users: single-channel guests free, multi-channel guests count as paid seats
- Support: email and chat support, response within 1-2 business days

**Annual savings:** Switching from monthly to annual billing saves approximately 17% per user per year.

---

### 2.3 Business+ Plan — $18/user/month (billed monthly) or $15/user/month (billed annually)

**Best for:** Mid-sized organizations that need security controls, compliance features, and advanced admin capabilities.

**Everything in Pro, plus:**
- Security: SAML-based Single Sign-On (SSO) — integrates with Okta, Microsoft Entra ID (Azure AD), OneLogin, and others
- User provisioning: SCIM (automatic user creation/deactivation synced with your identity provider)
- Compliance: full message and file export for legal discovery and compliance auditing (all channels including private, all DMs)
- Admin analytics: usage data, member activity reports, API access to admin data
- Uptime SLA: 99.99% uptime guarantee (versus best-effort on Free/Pro)
- AI features: advanced — AI-powered search across all Lumo content, daily channel recaps, message translation (40+ languages), file summaries, AI workflow generation
- Data Loss Prevention: integration with DLP tools
- Support: priority support, faster response times than Pro

**Important note on SSO:** Google Workspace SSO (sign in with Google) is available on ALL plans including Free. SAML-based SSO for enterprise identity providers like Okta or Microsoft Entra ID requires Business+ or higher.

**Common upgrade trigger:** IT team requires SSO via Okta/Entra; compliance team needs message export; organization needs to enforce 2FA across all members.

---

### 2.4 Enterprise Grid — Custom Pricing (typically $20–35+/user/month)

**Best for:** Large organizations (typically 500+ members) with complex governance, security, and cross-team coordination needs.

**Everything in Business+, plus:**
- Multiple connected workspaces under one org (e.g., separate workspaces for each business unit or region, managed centrally)
- Data residency: choose where your data is stored (US, EU, or Japan)
- HIPAA compliance: Business Associate Agreement (BAA) available — required for healthcare organizations handling PHI
- eDiscovery: advanced search and export tools for legal holds
- Enterprise Mobility Management (EMM): integration with MDM tools for mobile device policy enforcement
- Dedicated account manager
- Premium support: 4-hour response SLA for critical issues, 24/7 for P1 outages
- Custom data retention policies: set different retention periods per channel or workspace
- Cross-organizational analytics
- Professional services: onboarding, training, change management (quoted separately)

**Who needs Enterprise Grid:** Organizations with multiple business units that need separate workspaces but unified billing/governance; regulated industries (finance, healthcare, legal, government); organizations with strict data sovereignty requirements.

---

## 3. Feature Reference

### 3.1 Channels

The core organizational unit of Lumo. Channels can be public (anyone in the workspace can find and join) or private (invite-only). Every channel can have a canvas — a persistent collaborative document attached to the channel, available on all plans.

**Common support issues with channels:**
- Can't find a channel (may be private, or member hasn't been invited)
- Can't post in a channel (may be read-only/announcement channel — admins can restrict posting)
- Channel history missing (Free plan 90-day limit, or downgrade from paid to Free)
- Accidentally archived a channel (admin can unarchive from channel settings)

---

### 3.2 Direct Messages (DMs)

1:1 or small group conversations outside of channels. DMs do not have a "leave" option — they persist in the sidebar. You can mute them but not delete the conversation entirely.

**Common support issues:**
- Can't DM someone (they may be a single-channel guest restricted to one channel, or their DMs are restricted by admin)
- DM to deactivated account — messages show as sent but recipient can't see them
- Group DM not showing shared history (if a member was added later, they see history from when they joined)

---

### 3.3 Huddles

Real-time audio/video collaboration. Click the headphone icon in any channel or DM to start a huddle. Supports screen sharing, drawing on screen, AI-generated notes.

**Plan differences:**
- Free: 1:1 huddles only
- Pro+: group huddles up to 50 participants

**Common support issues:**
- Audio not working (microphone permission not granted to Lumo in OS settings)
- Noise cancellation not working on some hardware configurations (workaround: disable in Audio & Video preferences)
- Can't start huddle in a channel (may be Free plan, or admin has restricted huddles in that channel)
- Video not showing (camera permission not granted, or conflicting with another app like Zoom using camera)

---

### 3.4 Workflow Builder

No-code automation tool. Build workflows triggered by events (someone joins a channel, a form is submitted, a scheduled time) that send messages, create to-dos, collect information.

**Plan differences:**
- Free: up to 10 active workflows
- Pro+: unlimited workflows

**Common support issues:**
- Workflow not triggering (check that workflow is published, not just saved as draft)
- Workflow stopped working after member left (workflow may be owned by deactivated member — needs to be transferred)
- Form responses not showing up (check the channel the responses are being sent to)
- Conditional branching not working (requires Business+ new version of Workflow Builder)

---

### 3.5 Lumo AI

AI features built into the platform. Introduced gradually across plans.

**What's available by plan:**

| AI Feature | Free | Pro | Business+ | Enterprise |
|---|---|---|---|---|
| Thread/channel summaries | No | Basic | Advanced | Advanced |
| Huddle notes (AI) | No | Yes | Yes | Yes |
| AI-powered search | No | No | Yes | Yes |
| Daily recaps | No | No | Yes | Yes |
| Message translation | No | No | Yes | Yes |
| File summaries | No | No | Yes | Yes |
| AI workflow generation | No | No | Yes | Yes |

**Important pricing history:** Until June 2025, Lumo sold a separate AI add-on for $10/user/month. This was discontinued. Basic AI is now bundled into Pro. Advanced AI is bundled into Business+. Customers who had the old add-on were grandfathered until their next renewal, after which they need to choose a plan that includes the AI level they need.

**Common support issues:**
- "Where did my AI add-on go?" — discontinued June 2025, now bundled into plan tiers
- AI summaries are inaccurate — not a bug, AI summarization is best-effort
- AI search not finding things — requires Business+ plan; also may take 24–48 hours to index newly added content
- Can't turn off AI features — admin can disable AI for the workspace in Settings > Administration

---

### 3.6 Integrations & App Directory

Lumo's App Directory contains 2,400+ integrations. Popular ones include:

- **Productivity:** Google Drive, Dropbox, Box, Notion, Confluence
- **Project management:** Jira, Asana, Linear, Trello, Monday.com
- **Communication:** Zoom, Google Meet, Webex
- **CRM/Sales:** Salesforce, HubSpot
- **Developer tools:** GitHub, GitLab, PagerDuty, Datadog
- **HR:** Workday, BambooHR

**Plan differences:**
- Free: maximum 10 active integrations at a time
- Pro+: unlimited integrations

**Business+ admin controls:** Admins can require approval before members install new apps. If enabled, members see "Request to install" instead of "Add to Lumo."

**Common support issues:**
- Integration stopped working after password change (re-authorization required)
- Zoom/Google Meet integration posting duplicate notifications (known issue — fix: remove and re-add the integration)
- Can't install app (admin approval required, or workspace is at 10-app Free plan limit)
- App posting to wrong channel (check integration settings for notification channel)

---

### 3.7 Lumo Connect

Allows collaboration with people outside your organization. Two modes:

- **Channel Connect:** Share a channel with an external organization. Both sides can post. Requires both organizations to be on paid plans.
- **DM Connect:** Send direct messages to people at other organizations.

**Pricing note:** Lumo Connect usage may result in additional charges in some configurations — this is a common source of mystery line items on invoices.

**Common support issues:**
- External users can't see messages (their organization may not have accepted the Connect invite)
- Connect channel showing as disconnected (one side may have left or their plan expired)
- Unexpected Connect charge on invoice

---

### 3.8 Search

Full-text search across messages, files, and channels.

**Plan differences:**
- Free: searches only last 90 days of messages
- Pro+: searches unlimited history
- Business+: AI-powered search across all content including files

**Useful search modifiers:**
- `from:@person` — messages from a specific person
- `in:#channel` — messages in a specific channel
- `before:2024-06-01` / `after:2024-01-01` — date ranges
- `has:file` — messages with attachments

**Common support issues:**
- Can't find old messages (Free plan 90-day limit, or wrong date range filter applied)
- Search results missing a channel (private channel — only members can search it)
- Search indexing lag — new messages may take up to 24–48 hours to appear in search results

---

### 3.9 Notifications

Lumo sends notifications through desktop app banners, mobile push notifications, email digests, and in-app badges.

**Notification levels (per channel):**
1. All new messages (noisiest)
2. Direct messages, mentions & keywords (default — recommended)
3. Just @mentions (quietest for active workspaces)
4. Nothing (mute channel entirely)

**Common support issues:**
- Getting too many email notifications — check Preferences > Notifications > Email (default is "as soon as possible" when offline)
- Not getting any notifications — check OS notification permissions for Lumo, check Do Not Disturb schedule
- Mobile and desktop both notifying for same message — by design, configurable in preferences
- Notifications for a muted channel — check if a keyword alert matches messages in that channel

---

## 4. User Roles & Guest Policy

### 4.1 Roles

| Role | Capabilities |
|---|---|
| Owner | Full control — billing, delete workspace, transfer ownership. Only 1 owner per workspace. |
| Admin | Manage members, channels, integrations, settings. Cannot delete workspace or change billing without owner access. |
| Member | Regular user — can post, create channels (if allowed), install apps (if not restricted by admin). |
| Single-channel Guest | Can only see and post in one specific channel. FREE on paid plans. Cannot be made admin. |
| Multi-channel Guest | Can see and post in multiple channels. Counts as a FULL PAID SEAT. |

### 4.2 Deactivation vs Deletion

- Deactivating a member removes their access but preserves all their messages, files, and history
- Members cannot be permanently deleted from a workspace — only deactivated
- Deactivated members can be reactivated by an admin at any time
- Deactivated members' DMs become read-only to the other party

### 4.3 Ownership Transfer

- Current owner initiates in Settings > Administration > Workspace Settings > Transfer Ownership
- New owner receives email to confirm — both must confirm for transfer to complete
- If the original owner has left the company and cannot confirm: Lumo Support can facilitate transfer with proof of company domain ownership (DNS TXT record verification or business registration document)

---

## 5. Data & Compliance

### 5.1 Data Retention

| Plan | Message Storage | File Storage | After Cancellation |
|---|---|---|---|
| Free | 90 days searchable, 1 year before permanent deletion | 5 GB total | 90 days before workspace deleted |
| Pro | Unlimited | 10 GB/member | 90 days before workspace deleted |
| Business+ | Unlimited + configurable retention policies | 10 GB/member | 90 days before workspace deleted |
| Enterprise Grid | Unlimited + custom per-channel retention | Unlimited | Custom retention policies apply |

### 5.2 Export Capabilities

| Plan | What Can Be Exported |
|---|---|
| Free/Pro | Public channel messages and files only |
| Business+ | All messages (public, private, DMs) and files — full compliance export |
| Enterprise Grid | Everything in Business+ plus legal hold and eDiscovery tools |

**Export format:** JSON. Third-party tools can convert to CSV or other formats.

**How to initiate export:** Workspace Owner or Admin > Settings > Administration > Workspace Settings > Import/Export Data.

### 5.3 Compliance Certifications

- SOC 2 Type II: all plans
- ISO 27001: all plans
- GDPR: all plans (Lumo is a data processor; customer is data controller)
- HIPAA / BAA: Enterprise Grid only
- FedRAMP: Enterprise Grid (in progress as of 2026)
- CCPA: all plans

### 5.4 Data Residency

Enterprise Grid only. Options: United States (default), European Union, Japan.

---

## 6. Support Structure

### 6.1 Support Channels by Plan

| Plan | Support Options | Response Time |
|---|---|---|
| Free | Help Center, community forums, in-app guided setup | No direct support |
| Pro | Email and chat support | 1–2 business days |
| Business+ | Priority email and chat support | Faster than Pro, dedicated queue |
| Enterprise Grid | Dedicated account manager, 4-hour SLA for critical issues, 24/7 for P1 | 4 hours (critical) |

**Note:** Lumo does not offer phone support on any plan.

### 6.2 Support Escalation Path

1. Self-service: Help Center at help.lumo.com
2. Chat support: available from any Lumo page, bottom-right corner
3. Email ticket: support@lumo.com (include workspace URL and issue description)
4. Enterprise Grid: contact your dedicated account manager directly

### 6.3 What Support Can and Cannot Do

**Support CAN:**
- Issue refunds within policy
- Resend invoices to billing email on file
- Reset 2FA for locked-out members (with identity verification)
- Facilitate workspace ownership transfers with proof of company ownership
- Extend trials up to 30 additional days for qualified accounts
- Apply promotional credits to accounts

**Support CANNOT:**
- Access or read your workspace messages — Lumo support has no access to workspace content
- Issue refunds outside the 30-day window except in documented billing error cases
- Create accounts on behalf of users
- Force password changes — must go through standard reset flow
- Change billing plan without the workspace owner's authorization
