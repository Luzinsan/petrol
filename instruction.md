Тренды АСУТП.zip на Drive: https://drive.google.com/file/d/12pMELyLcCTwKZL_9vg7hGKTajWidAHj_/view?usp=drive_link

{
  "access_token": "ya29.a0Ad52N39Kd1lhrPgU4BeSDVOT9dbz3bpxGIaM6vglsVnpaSx-WnC41sD5XOaJB4Fek7MyybjkfU_fJPIyX9rOI6sLEgplqBEubEK4CKCI8Fg3uOjY5TK2o-9ca5ZdEvyzsLCupVTRXduYxypqmfnWnfCAS8nIqmGC0llcaCgYKAXESARASFQHGX2Mip0DC2CqfBg1CZYWyh5IDFw0171", 
  "scope": "https://www.googleapis.com/auth/drive.readonly", 
  "token_type": "Bearer", 
  "expires_in": 3599, 
  "refresh_token": "1//04fLRkh3zDLLzCgYIARAAGAQSNwF-L9IrA6JaXvEk3-I8p1P9C4J5G38H6QSsX2N6QGG_jIJDkaXlwDdY6vEkZCQ2D8mwyKzNlpA"
}

Для скачивания напрямую с Drive: curl -H "Authorization: Bearer ya29.a0Ad52N39Kd1lhrPgU4BeSDVOT9dbz3bpxGIaM6vglsVnpaSx-WnC41sD5XOaJB4Fek7MyybjkfU_fJPIyX9rOI6sLEgplqBEubEK4CKCI8Fg3uOjY5TK2o-9ca5ZdEvyzsLCupVTRXduYxypqmfnWnfCAS8nIqmGC0llcaCgYKAXESARASFQHGX2Mip0DC2CqfBg1CZYWyh5IDFw0171" https://www.googleapis.com/drive/v3/files/12pMELyLcCTwKZL_9vg7hGKTajWidAHj_?alt=media -o asutp.zip 
(Инструкция по получению ссылки выше: https://www.quora.com/How-do-I-download-a-very-large-file-from-Google-Drive)

---

Установка git-lfs: https://packagecloud.io/github/git-lfs/install#bash-deb
Далее: 
$ sudo apt-get install git-lfs
$ git lfs install
Как только в репозитории появится .zip, выполняется (https://github.com/git-lfs/git-lfs/issues/325):
$ git lfs fetch