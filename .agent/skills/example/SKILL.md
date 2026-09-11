---
name: example
description: Contoh skill - panduan untuk tugas umum
---

# Example Skill

Skill ini adalah contoh bagaimana agent dapat menggunakan panduan dari `.agent/skills`.

## Kapan digunakan
- Saat user meminta contoh penggunaan skill
- Untuk demonstrasi loading skill via `skill_load`

## Instruksi untuk Agent
1. Jika user bertanya tentang skill, panggil `skill_list` untuk melihat katalog.
2. Panggil `skill_load` dengan nama skill untuk mendapatkan detail.
3. Ikuti panduan di SKILL.md ini langkah demi langkah.
4. Selalu verifikasi hasil dengan tool `read` / `bash` sebelum menjawab.

## Contoh workflow
1. `skill_list` -> lihat ada skill `example`
2. `skill_load` name=example -> baca panduan ini
3. Eksekusi tugas sesuai panduan

## Catatan
Tambahkan skill baru dengan membuat folder `.agent/skills/<nama>/SKILL.md`.
Format SKILL.md bebas, tapi usahakan sertakan `name` dan `description` di frontmatter.
