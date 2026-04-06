# [FEATURE] Authentication Module - Login, Logout, Register, Forgot Password, RBAC

## Labels
`feature`, `security`, `high-priority`

## Assignees
@xuanquangIT

---

## Description

Implement a complete authentication and authorization system for PV Lakehouse Dashboard.
Currently all pages and API endpoints are publicly accessible with zero authentication.

### Goals
- User login / logout with JWT session tokens (httpOnly cookie)
- Admin-only user registration
- Self-service password reset via SMTP email
- Role-Based Access Control (RBAC) for all 8 modules
- Protect all existing routes behind authentication

---

## RBAC Roles (5 total)

| Role | Display Name | Type | Interactive Login |
|------|-------------|------|-------------------|
| `admin` | Quan ly | Manager / Approver | Yes |
| `data_engineer` | Data Engineer | Pipeline Owner | Yes |
| `ml_engineer` | ML Engineer | Model Developer | Yes |
| `analyst` | Analyst | Data Consumer | Yes |
| `system` | System | Auto Scheduler | No |

## Permission Matrix

| Module | Quan ly | Data Engineer | ML Engineer | Analyst | System |
|--------|---------|---------------|-------------|---------|--------|
| Dashboard | Full | Full | Read | Read | -- |
| Data Pipeline | -- | Full | -- | -- | Auto |
| Data Quality | -- | Full | -- | -- | -- |
| ML Training | -- | -- | Full | -- | Auto |
| Model Registry | Approve | -- | Full | -- | Auto |
| Forecast | Read | Read | Read | Read | -- |
| Analytics | Read | -- | -- | Full | -- |
| Solar AI Chat | Full | Full | Full | -- | -- |

---

## Tasks

### Database
- [ ] Create `auth_roles` table with 5 seeded roles
- [ ] Create `auth_users` table with UUID PK, bcrypt password hash
- [ ] Create `auth_password_resets` table for email-based reset tokens
- [ ] Seed default admin user (`admin / admin123`)

### Backend Core
- [ ] Add `AuthSettings` class to settings.py (JWT config, cookie config)
- [ ] Add `SmtpSettings` class to settings.py (SMTP host, port, user, password, TLS)
- [ ] Create `security.py` (password hash/verify, JWT create/decode, reset token gen)
- [ ] Create `database.py` (SQLAlchemy engine, session factory, `get_db()` dependency)
- [ ] Create `email.py` (send password reset email via SMTP in background thread)

### Backend Schemas
- [ ] `LoginRequest`, `UserRead`, `UserCreate`
- [ ] `TokenPayload`, `ForgotPasswordRequest`, `ResetPasswordRequest`

### Backend Repository
- [ ] `user_repository.py` (get by username/id/email, create, list, update password)
- [ ] `reset_repository.py` (create token, get valid, mark used)

### Backend Service
- [ ] `auth_service.py` (authenticate, register, token creation, password reset flow)

### Backend API
- [ ] `GET/POST /auth/login` -- login page and form handler
- [ ] `POST /auth/logout` -- clear session cookie
- [ ] `GET /auth/me` -- current user info (JSON)
- [ ] `GET/POST /auth/register` -- admin-only user creation
- [ ] `GET/POST /auth/forgot-password` -- request password reset email
- [ ] `GET/POST /auth/reset-password/{token}` -- set new password
- [ ] `dependencies.py` -- `get_current_user()` and `require_role()` middleware

### Frontend Templates
- [ ] `login.html` -- standalone login page with glassmorphism design
- [ ] `register.html` -- admin-only registration form (extends base.html)
- [ ] `forgot_password.html` -- email input for password reset
- [ ] `reset_password.html` -- new password form with token validation
- [ ] `auth.css` -- styles for all auth pages
- [ ] Update `base.html` -- user info in header, logout button, role-based nav

### Integration
- [ ] Register `auth.router` in `main.py`
- [ ] Add auth middleware to inject `request.state.user`
- [ ] Protect existing routes in `frontend.py` with `get_current_user`
- [ ] Apply `require_role()` to all module API routers

### Configuration
- [ ] Update `.env_example` with auth + SMTP variables
- [ ] Update `.env` with dev defaults
- [ ] Add `passlib[bcrypt]`, `python-jose[cryptography]`, `python-multipart` to requirements.txt

### Testing
- [ ] Unit tests for security utilities
- [ ] Integration tests for auth API endpoints
- [ ] RBAC permission tests per role
- [ ] Browser validation of full auth flow

---

## Technical Decisions
- **Session**: Stateless JWT in httpOnly cookie
- **Password hashing**: bcrypt via passlib
- **Token library**: python-jose
- **Email**: Python stdlib smtplib (no extra dependency)
- **Architecture**: api -> services -> repositories -> PostgreSQL

## Acceptance Criteria
1. Unauthenticated users are redirected to `/auth/login`
2. Login with valid credentials sets httpOnly cookie and redirects to `/`
3. Header displays logged-in username and role badge
4. Logout clears cookie and redirects to login page
5. Admin can register new users with any of the 4 interactive roles
6. Users can reset their password via email link
7. Each role can only access modules permitted by the RBAC matrix
8. Unpermitted module access returns HTTP 403

## Files Affected
17 new files, 8 modified files -- see implementation plan for full list.
