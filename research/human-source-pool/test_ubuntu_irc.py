import tempfile
import unittest
from pathlib import Path
from ingest_ubuntu_irc import channel_ok, parsed_runs, english_prose, make_pair, SID
from expand_pool import connect,add


def human_text():
    return '\n'.join('[12:%02d] <person%d> I need some help with the computer and the program because it does not work for my account %d.'%(i,i%3,i) for i in range(8))


def source_row(text=None):
    return {'id':'2008-01-01-#ubuntu','created':'2008-01-01','text':text or human_text(),
            'metadata':{'channel':'#ubuntu','license':'Public Domain','url':'https://irclogs.ubuntu.com/2008/01/01/%23ubuntu.txt'}}


class UbuntuIRCTests(unittest.TestCase):
    def test_bot_and_system_lines_break_contiguous_runs(self):
        text='[12:00] <alice> I need some help with this computer\n[12:01] <ubottu> I can show you how to get help\n[12:02] <bob> I need help with another program\n[12:03] *** bob has joined\n[12:04] <charlie> I also have a question'
        runs=parsed_runs(text)
        self.assertEqual(len(runs),3)
        self.assertEqual([r[0][3] for r in runs],['alice','bob','charlie'])
        for run in runs:
            value=text[run[0][0]:run[-1][1]]
            self.assertNotIn('ubottu',value);self.assertNotIn('***',value)

    def test_long_time_gaps_and_repeated_bot_bodies_split(self):
        line='[12:00] <unknownservice> This is an automated update about a ticket in the tracking service\n'
        self.assertEqual(parsed_runs(line*4),[])
        self.assertEqual(len(parsed_runs('[12:00] <a> I need help\n[13:00] <b> I can help')),2)

    def test_language_channels_and_text(self):
        self.assertTrue(channel_ok('#ubuntu-us-or'));self.assertFalse(channel_ok('#ubuntu-ir'));self.assertFalse(channel_ok('#ubuntu-fr'))
        self.assertTrue(english_prose([human_text()]))
        self.assertFalse(english_prose(['bonjour salut merci ordinateur probleme question francais '*20]))
        self.assertFalse(english_prose(['hello the and you '+'سلام '*100]))

    def test_exact_offsets_tags_speaker_family_and_quarantine(self):
        original=source_row();pair=make_pair(original,'raw/documents/00000_ubuntu.jsonl.gz',5,[10,10,0,0],'hash')
        self.assertIsNotNone(pair);row,raw=pair
        self.assertEqual(original['text'][row['raw_start']:row['raw_end']],row['text'])
        self.assertEqual(row['word_count'],len(row['text'].split()))
        self.assertEqual(row['source_id'],SID);self.assertEqual(row['category'],'social')
        self.assertTrue(row['speaker_family_ids']);self.assertFalse(row['training_eligible'])
        self.assertEqual(raw['record'],original)

    def test_date_license_and_channel_restrictions(self):
        for field,value in [('created','2025-01-01'),('license','unknown'),('channel','#ubuntu-de')]:
            row=source_row()
            if field=='created':row[field]=value
            else:row['metadata'][field]=value
            self.assertIsNone(make_pair(row,'shard',0,[9]*4,'hash'))

    def test_dedup_and_quota_across_resume(self):
        pair=make_pair(source_row(),'shard',0,[9,0,0,0],'hash')
        with tempfile.TemporaryDirectory() as tmp:
            dbpath=Path(tmp)/'db';db=connect(dbpath)
            with db:self.assertTrue(add(db,*pair,1));self.assertFalse(add(db,*pair,1))
            db.close();db=connect(dbpath)
            with db:self.assertFalse(add(db,*pair,1))
            db.close()


if __name__=='__main__':unittest.main()
