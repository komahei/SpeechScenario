# SpeechScenario
音声合成を用いたシナリオ支援アプリ

#実行動画



[![支援アプリ動画](https://img.youtube.com/vi/3ilFax8-2Bw/0.jpg)](https://www.youtube.com/watch?v=3ilFax8-2Bw)

#使用させていただいたもの
tts_implementationにて実装されている音声合成のコードはPythonで学ぶ音声合成機械学習実践シリーズからTacotron2及びParallelWaveGANのコードを使用させていただいております。
https://github.com/r9y9/ttslearn
https://book.impress.co.jp/books/1120101073
話者特徴量をtacotron2のEncoderにconcatすることで音声データから特定話者の音声を合成できるよう変更していますが，話者特徴量の抽出にはx-vectorの学習済みモデルを使用させていただいております。
https://github.com/sarulab-speech/xvector_jtubespeech
