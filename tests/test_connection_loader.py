import base64
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == 'nt', 'actual Windows connection loader')
class WindowsConnectionLoaderTests(unittest.TestCase):
    def test_research_endpoints_are_literal_typed_and_numbered(self):
        loader = Path(__file__).resolve().parents[1]/'deploy/windows/ReadBotConnections.ps1'
        with tempfile.TemporaryDirectory() as temp:
            connections = Path(temp)/'test.env'
            url = 'https://rpc.invalid/rpc?apikey=fixture&other=a=b'
            connections.write_text('SOLANA_RESEARCH_RPC_URL='+url+'\nSOLANA_RESEARCH_WS_URL_300=wss://rpc.invalid/ws\n', encoding='utf-8')
            script = f". '{loader}'; Import-BotConnections -Path '{connections}'; if($env:SOLANA_RESEARCH_RPC_URL -ne '{url}' -or $env:SOLANA_RESEARCH_WS_URL_300 -ne 'wss://rpc.invalid/ws'){{exit 1}}; 'ENDPOINTS_LOADED'"
            result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode()],capture_output=True,text=True,timeout=30,check=True)
            self.assertIn('ENDPOINTS_LOADED', result.stdout)
            for declaration in ('SOLANA_RESEARCH_RPC_URL=http://remote.invalid/?key=must-not-print',
                                'SOLANA_RESEARCH_WS_URL=https://remote.invalid',
                                'SOLANA_RESEARCH_RPC_URL_0=https://remote.invalid',
                                'SOLANA_RESEARCH_RPC_URL=https://remote.invalid/#fragment'):
                connections.write_text(declaration+'\n', encoding='utf-8')
                script = f". '{loader}'; try{{Import-BotConnections -Path '{connections}';exit 1}}catch{{'REFUSED_ENDPOINT';exit 0}}"
                result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode()],capture_output=True,text=True,timeout=30,check=True)
                self.assertIn('REFUSED_ENDPOINT', result.stdout)
                self.assertNotIn('must-not-print', result.stdout+result.stderr)

    def test_credentials_are_literal_and_cannot_enable_live_signing(self):
        loader = Path(__file__).resolve().parents[1]/'deploy/windows/ReadBotConnections.ps1'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root/'must-not-exist'
            connections = root/'test.env'
            literal = f'$(Set-Content -LiteralPath "{marker}" -Value unsafe)'
            connections.write_text('HELIUS_API_KEY=fixture-key-not-real\nYELLOWSTONE_GRPC_TOKEN='+literal+'\n',encoding='utf-8')
            script = f". '{loader}'; Import-BotConnections -Path '{connections}'; @{{key_loaded=($env:HELIUS_API_KEY -eq 'fixture-key-not-real');token_literal=($env:YELLOWSTONE_GRPC_TOKEN.StartsWith('$('));marker_created=(Test-Path -LiteralPath '{marker}')}}|ConvertTo-Json -Compress"
            result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode()],capture_output=True,text=True,timeout=30,check=True)
            proof = json.loads(result.stdout)
            self.assertTrue(proof['key_loaded'])
            self.assertTrue(proof['token_literal'])
            self.assertFalse(proof['marker_created'])
            connections.write_text('ALLOW_LIVE_TRADING=yes-i-understand\n',encoding='utf-8')
            script = f". '{loader}'; try{{Import-BotConnections -Path '{connections}';exit 1}}catch{{'REFUSED_UNSUPPORTED_FIELD';exit 0}}"
            result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode()],capture_output=True,text=True,timeout=30,check=True)
            self.assertIn('REFUSED_UNSUPPORTED_FIELD',result.stdout)
