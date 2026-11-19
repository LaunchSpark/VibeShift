# NRC EmoLex setup

EmoLex is provided by National Research Council Canada for non-commercial
research and educational use. Its terms prohibit redistribution, so the archive
and extracted text file must stay local. Review the terms on the [official
EmoLex page](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), then
download and install the lexicon with:

```powershell
python -m evaluation.models.nrc_lexicon.download_lexicon
```

The script retains only `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt` in the ignored
`data/` folder. If you downloaded the archive in a browser, the alternate
`prepare_lexicon.py <archive.zip>` helper extracts that same file. The runner
drops trust and anticipation, uses equal token counts over the target plus three preceding messages for
`nrc_wordcount`, and uses only the preceding 25 messages for `nrc_macro`.

Please cite Mohammad and Turney's NRC Emotion Lexicon papers when reporting
results. The official terms and citation details are on the [NRC access
page](https://saifmohammad.com/WebPages/AccessResource.htm).
