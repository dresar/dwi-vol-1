# Spesifikasi Desain Halaman — Sistem Rekomendasi Musim Tanam (Desktop-first)

## Global Styles (berlaku untuk semua halaman)
- Layout system: kombinasi CSS Grid (struktur halaman) + Flexbox (komponen).
- Breakpoints: Desktop ≥ 1024px (utama), Tablet 768–1023px, Mobile ≤ 767px.
- Design tokens:
  - Background: #0B1220 (app shell) / #0F172A (panel)
  - Surface/Card: #111C33
  - Text primary: #E5E7EB, secondary: #94A3B8
  - Primary brand: #22C55E (aksi utama “Buat Rekomendasi”)
  - Danger: #EF4444, Warning: #F59E0B
  - Border: rgba(148,163,184,0.18)
  - Radius: 12px (card), 10px (input)
  - Shadow: 0 12px 30px rgba(0,0,0,0.35)
- Typography:
  - H1 28/34 semibold, H2 20/28 semibold, Body 14/22 regular
- Buttons:
  - Primary: bg #22C55E, text #052E16; hover: #16A34A; disabled: opacity 0.5
  - Secondary: bg transparent, border 1px; hover: bg rgba(148,163,184,0.08)
- Inputs:
  - Default: border 1px; focus ring 2px (rgba(34,197,94,0.35))
  - Error: border #EF4444 + helper text
- Table:
  - Header sticky (desktop), zebra row halus, kolom aksi rata kanan
- Motion:
  - Transisi 160–220ms untuk hover/focus; skeleton loading untuk hasil rekomendasi

## 1) Halaman Login
### Meta Information
- Title: "Login — Rekomendasi Musim Tanam"
- Description: "Masuk untuk membuat dan melihat rekomendasi musim tanam berbasis data."
- Open Graph: og:title, og:description, og:type=website

### Layout
- 2 kolom desktop (Grid 12 kolom): kiri branding/ilustrasi (5), kanan form login (7).
- Tablet/mobile: stacked (branding di atas, form di bawah).

### Page Structure
1. Brand Panel (kiri)
2. Login Card (kanan)
3. Footer mini (help text)

### Sections & Components
- Brand Panel
  - Logo + nama sistem
  - Copy singkat: “Rekomendasi musim tanam berbasis Random Forest + penjelasan SHAP.”
- Login Card
  - Judul: “Masuk”
  - Form:
    - Input Email/Username
    - Input Password + toggle show/hide
    - Tombol “Masuk” (primary)
    - Pesan error ter-atas (alert) dan error per-field
  - Link kecil (opsional): “Lupa password” (jika fitur ada nanti; jika tidak, sembunyikan)
- State
  - Loading: tombol disabled + spinner kecil

## 2) Halaman Dashboard (Role-based)
### Meta Information
- Title: "Dashboard — Rekomendasi Musim Tanam"
- Description: "Ringkasan dan akses cepat ke rekomendasi, riwayat, dan manajemen model."

### Layout
- App shell: sidebar kiri (240px) + main content.
- Sidebar collapsible di tablet; jadi topbar + drawer di mobile.

### Page Structure
1. Sidebar Navigasi
2. Topbar (judul halaman + user menu)
3. Content grid (kartu ringkasan + aktivitas terbaru)

### Sections & Components
- Sidebar
  - Menu: Dashboard, Buat Rekomendasi, Riwayat
  - Admin-only: Manajemen Data & Model
  - Tombol Logout (bagian bawah)
- Topbar
  - Breadcrumb opsional
  - User menu: nama + role badge (ADMIN/PETANI)
- Konten (Petani)
  - Kartu: “Buat Rekomendasi Baru” (CTA utama)
  - Kartu: “Rekomendasi Terakhir” (tanggal, lokasi, hasil)
  - Panel: “Riwayat Terbaru” (3–5 item)
- Konten (Admin)
  - Kartu: “Model Aktif” (versi, tanggal)
  - Kartu: “Metrik Terakhir” (ringkas)
  - Kartu: “Dataset Terbaru” (nama file, row count)
  - Panel: “Audit Aktivitas Terakhir”

## 3) Halaman Buat Rekomendasi
### Meta Information
- Title: "Buat Rekomendasi — Rekomendasi Musim Tanam"
- Description: "Masukkan kondisi lahan/iklim untuk mendapatkan rekomendasi musim tanam dan penjelasannya."

### Layout
- Desktop: 2 kolom (Form 5/12, Hasil 7/12).
- Jika belum ada hasil: kolom hasil menampilkan empty state.

### Page Structure
1. Panel Form Input
2. Panel Hasil Rekomendasi
3. Panel Penjelasan SHAP (di bawah hasil atau tab)

### Sections & Components
- Form Input (Card)
  - Fieldset “Kondisi Lokasi”
    - Lokasi (text / select)
  - Fieldset “Tanah”
    - Jenis tanah (select)
    - pH tanah (number)
  - Fieldset “Iklim”
    - Curah hujan (number)
    - Suhu (number)
    - Kelembapan (number)
  - Fieldset “Lahan & Komoditas”
    - Luas lahan (number)
    - Komoditas (optional)
  - Tombol aksi:
    - “Dapatkan Rekomendasi” (primary)
    - “Reset” (secondary)
- Panel Hasil (Card)
  - Header: badge versi model + timestamp
  - Hasil utama: “Rekomendasi Musim Tanam: …” (teks besar)
  - Score/confidence (jika ada): progress bar atau chip
  - Tombol: “Simpan ke Riwayat”
- Panel SHAP
  - Tab: “Ringkasan” | “Detail”
  - Ringkasan:
    - Grafik bar horizontal Top-8 fitur (positif/negatif berbeda warna)
    - Paragraf interpretasi singkat (template teks)
  - Detail:
    - Tabel: feature, nilai input, shap_value
- States
  - Loading: skeleton di panel hasil + disable tombol
  - Error: alert “Model belum aktif / input tidak valid / layanan gagal”

## 4) Halaman Riwayat Rekomendasi
### Meta Information
- Title: "Riwayat — Rekomendasi Musim Tanam"
- Description: "Lihat rekomendasi yang pernah dibuat beserta detail dan penjelasan SHAP."

### Layout
- Desktop: table view utama + drawer/side panel untuk detail.
- Mobile: list cards + detail halaman penuh.

### Page Structure
1. Toolbar filter
2. Tabel/List riwayat
3. Detail rekomendasi (drawer/modal/side panel)

### Sections & Components
- Toolbar
  - Pencarian (lokasi/komoditas)
  - Filter tanggal (range sederhana)
- Tabel
  - Kolom: tanggal, lokasi, komoditas, hasil, versi model, aksi “Detail”
- Detail Panel
  - Ringkasan input (grid 2 kolom)
  - Output rekomendasi
  - SHAP ringkas (mini bar chart + top-5)

## 5) Halaman Admin: Manajemen Data & Model
### Meta Information
- Title: "Manajemen Data & Model — Admin"
- Description: "Kelola dataset, latih model Random Forest, evaluasi, dan aktivasi versi model."

### Layout
- Desktop: tabbed admin workspace.
- Tab: Dataset | Training | Model Versions | Audit

### Page Structure
1. Header + deskripsi singkat
2. Tabs
3. Konten tab

### Sections & Components
- Tab Dataset
  - Upload card
    - Dropzone CSV
    - Info kolom wajib (ditampilkan sebagai checklist)
  - Dataset table
    - filename, created_at, row_count, status validasi
- Tab Training
  - Pilih dataset (select)
  - Training config minimal (mis. n_estimators, max_depth) dengan default tersembunyi di “Advanced”
  - Tombol “Latih Model” + progress log ringkas
- Tab Model Versions
  - List versi model
    - versi, dataset sumber, metrik ringkas, created_at
    - aksi: “Aktifkan” (hanya satu aktif)
  - Panel detail metrik (grafik sederhana)
- Tab Audit
  - Tabel log: waktu, actor, action, ringkasan detail
- States & Guards
  - Admin-only route guard (jika role bukan ADMIN: tampilkan halaman 403 ringkas atau redirect ke Dashboard)
  - Konfirmasi dialog untuk “Aktifkan model” dan “Latih model”