---
title: "Feature: Tích hợp nhúng Dashboard Power BI (App-Owns-Data)"
assignees: []
labels: ["enhancement", "frontend", "backend", "powerbi"]
status: "Completed"
---

# Mô tả Issue
Yêu cầu tích hợp trực tiếp biểu đồ (Dashboard) từ Power BI lên giao diện của hệ thống PV Lakehouse. Quá trình nhúng phải tuân thủ chuẩn kiến trúc của dự án, sử dụng FastAPI cho backend và Jinja2/HTML cho frontend, đảm bảo người xem trên web không cần tài khoản Power BI Pro.

Tính năng nhúng phải đảm bảo giấu kín `CLIENT_SECRET` hoàn toàn ở tầng Backend, dùng thư viện Microsoft Authentication Library (MSAL) để tạo Auth Token.

# ✅ Những gì đã hoàn thành (Completed)

## Configuration & Environment
- [x] Thêm định nghĩa cấu hình biến môi trường `POWERBI_*` (Tenant ID, Client ID, Workspace ID, Report ID) vào file `.env_example`.
- [x] Integrate cấu hình Power BI vào `main/backend/app/core/settings.py` qua `PowerBISettings`.

## Backend
- [x] Thêm package `msal` vào `requirements.txt`.
- [x] Xây dựng Service `main/backend/app/services/powerbi_service.py` đảm nhận việc gửi yêu cầu lên Entra ID để tạo `access_token` và yêu cầu API Power BI xuất `embed_token` cùng `embed_url`.
- [x] Tạo endpoint bảo mật `GET /dashboard/embed-info` trong module `dashboard`.

## Frontend
- [x] Cập nhật điều hướng Header trong `base.html`.
- [x] Tạo template `main/frontend/templates/dashboard.html` sử dụng JsDelivr CDN của `powerbi-client`.
- [x] Tuỳ chỉnh CSS `page-shell` để khung nhúng hiển thị rộng 96% màn hình và cao 850px nhằm mang lại trải nghiệm xem tốt nhất giống như Native.
- [x] Cài đặt fallback UI "Mock Mode" hiển thị hộp thông báo lỗi khi cấu hình `.env` bị rỗng.

# 🚧 Những tính năng cần mở rộng (Follow-up Tasks)
- [ ] Tính năng **Scheduled/Manual Refresh Data từ Web**: Viết thêm endpoint `POST /dashboard/refresh` cho phép Website gửi lệnh ép buộc Power BI Server (API `/refreshes`) cập nhật dữ liệu tự động thay vì phải mở bằng tay.
- [ ] Bảo mật bổ sung: Phân quyền Authorization (RBAC) để xem Dashboard Endpoint này chỉ cho phép Users có quyền "Viewer" trở lên trong hồ sơ CSDL Postgres.

# Hướng dẫn cấu hình môi trường
Các môi trường (dev, stg, prod) phải có Azure App Configuration với quyền đọc (Report.Read.All, Workspace.Read.All) trước khi chạy lên production. 
Chi tiết tham khảo tài liệu `huong_dan_cai_dat_powerbi.md`.
