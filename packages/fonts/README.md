# hanji-fonts

리눅스·Docker처럼 한글 글꼴이 없는 환경에서, 글꼴을 임베드하지 않은 한글 PDF를 hanji가 읽을 수 있게 하는 대체 글꼴(Noto Sans KR Regular) 패키지.

- 설치: `pip install "hanji[fonts]"`
- 필요 없는 경우: macOS·Windows, 또는 시스템에 한글 글꼴이 이미 있는 리눅스
- 출처: `notofonts/noto-cjk` 태그 `Sans2.004`의 `Sans/SubsetOTF/KR/NotoSansKR-Regular.otf`
  (SHA-256 `69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68`)
- 라이선스: 글꼴은 SIL Open Font License 1.1(`src/hanji_fonts/fonts/OFL.txt`), 파이썬 코드는 Apache-2.0
- 알려진 한계: `∙` `‣` `▸` 글리프가 없다
- 고정 정책: 4.6MB 바이너리가 git에 들어가므로 글꼴 교체는 드물게 한다. 자주 바꿔야 하면 Git LFS를 검토한다
