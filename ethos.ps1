$host.UI.RawUI.WindowTitle = 'ethos'
$env:PYTHONPATH = "$PSScriptRoot;$env:PYTHONPATH"
& python -m ethos @args
