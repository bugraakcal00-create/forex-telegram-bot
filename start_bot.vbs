Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "c:\Users\MSI\Desktop\forex_telegram_bot"
WshShell.Run """c:\Users\MSI\Desktop\forex_telegram_bot\start_bot.bat""", 0, False
Set WshShell = Nothing
