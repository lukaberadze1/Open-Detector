; The Windows installer, compiled by deploy/windows/build_app.py with Inno Setup. It installs for the
; current user into %LOCALAPPDATA%\Programs, so it asks for no administrator password.

[Setup]
AppId={{6E1B7C52-3F0A-4D7E-9C1B-2A8F5D0E4B16}
AppName=Open Detector
AppVersion={#Version}
AppPublisher=Open Detector
AppPublisherURL=https://github.com/lukaberadze1/Open-Detector
DefaultDirName={autopf}\Open Detector
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputBaseFilename={#OutputName}
SetupIconFile=app.ico
UninstallDisplayIcon={app}\Open Detector.exe
Compression=lzma2
SolidCompression=yes
LZMANumBlockThreads=4
WizardStyle=modern

[Tasks]
Name: desktopicon; Description: "Add a shortcut to the desktop"

[InstallDelete]
; An update replaces the app whole, so nothing of an older version is left behind. What the app writes
; is in %LOCALAPPDATA%\Open Detector, not here.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\app"

[Files]
Source: "{#Stage}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Bootstrapper}"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: NeedsWebView2

[Icons]
Name: "{autoprograms}\Open Detector"; Filename: "{app}\Open Detector.exe"
Name: "{autodesktop}\Open Detector"; Filename: "{app}\Open Detector.exe"; Tasks: desktopicon

[Run]
Filename: "{tmp}\MicrosoftEdgeWebview2Setup.exe"; Parameters: "/silent /install"; StatusMsg: "Installing Microsoft WebView2..."; Check: NeedsWebView2
Filename: "{app}\Open Detector.exe"; Description: "Open Open Detector"; Flags: postinstall nowait skipifsilent

[Code]
const
  WebView2Key = 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  WebView2Key32 = 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

function Installed(Root: Integer; Key: String): Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(Root, Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0');
end;

function NeedsWebView2: Boolean;
begin
  Result := not (Installed(HKLM, WebView2Key32) or Installed(HKLM, WebView2Key) or Installed(HKCU, WebView2Key));
end;
