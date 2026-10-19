import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.text import email_text
class TestProject(unittest.TestCase):
 def test_api(self):
  with TestClient(app) as c:
   self.assertEqual(c.get('/health').json()['status'],'ok')
   self.assertIn('Антиспам',c.get('/').text)
   self.assertEqual(c.post('/predict',json={'text':''}).status_code,422)
   self.assertEqual(c.post('/predict',json={'text':'a'*50001}).status_code,422)
   r=c.post('/predict',json={'text':'Meeting agenda for the research project on Tuesday'})
   self.assertEqual(r.status_code,200)
   self.assertTrue(0<=r.json()['spam_score']<=1)
   self.assertIn(r.json()['label'],('ham','spam'))
   self.assertTrue(r.json()['category_experimental'])
 def test_mime_ignores_attachment(self):
  raw=b'Subject: Hello\nMIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=x\n\n--x\nContent-Type: text/plain\n\nmeeting\n--x\nContent-Type: text/plain\nContent-Disposition: attachment; filename="payload.txt"\n\nATTACHMENT_SECRET\n--x--\n'
  text=email_text(raw);self.assertIn('meeting',text);self.assertNotIn('ATTACHMENT_SECRET',text)
 def test_html_does_not_run_script(self):
  text=email_text(b'Subject: Hi\nContent-Type: text/html\n\n<p>Hello</p><script>SECRET_CODE</script>')
  self.assertIn('Hello',text);self.assertNotIn('SECRET_CODE',text)
if __name__=='__main__':unittest.main(verbosity=2)
