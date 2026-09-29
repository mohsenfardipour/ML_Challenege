import tempfile
import unittest
from pathlib import Path

import organizer


class OrganizerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, name, text="x"):
        p = self.root / name
        p.write_text(text)
        return p

    def test_categories(self):
        for name, folder in [("a.JPG", "Images"), ("a.mkv", "Videos"), ("a.flac", "Music"),
                             ("a.pdf", "Documents"), ("a.7z", "Archives"),
                             ("a.msi", "Installers"), ("a.xyz", "Others"), ("noext", "Others")]:
            self.assertEqual(organizer.category_for(Path(name)), folder, name)

    def test_organize_moves_files(self):
        self.make("photo.png"); self.make("song.mp3"); self.make("weird.abc")
        moves = organizer.organize_folder(self.root, log=lambda m: None)
        self.assertEqual(len(moves), 3)
        self.assertTrue((self.root / "Images" / "photo.png").exists())
        self.assertTrue((self.root / "Music" / "song.mp3").exists())
        self.assertTrue((self.root / "Others" / "weird.abc").exists())

    def test_never_overwrites(self):
        (self.root / "Documents").mkdir()
        (self.root / "Documents" / "a.txt").write_text("old")
        self.make("a.txt", "new")
        organizer.organize_folder(self.root, log=lambda m: None)
        self.assertEqual((self.root / "Documents" / "a.txt").read_text(), "old")
        self.assertEqual((self.root / "Documents" / "a (1).txt").read_text(), "new")

    def test_partial_files_untouched(self):
        for n in ("x.crdownload", "y.part", "z.tmp"):
            self.make(n)
        moves = organizer.organize_folder(self.root, log=lambda m: None)
        self.assertEqual(moves, [])
        for n in ("x.crdownload", "y.part", "z.tmp"):
            self.assertTrue((self.root / n).exists())

    def test_subfolders_untouched(self):
        sub = self.root / "myfolder"; sub.mkdir()
        (sub / "inner.jpg").write_text("x")
        organizer.organize_folder(self.root, log=lambda m: None)
        self.assertTrue((sub / "inner.jpg").exists())

    def test_undo_restores_and_deletes_nothing(self):
        self.make("a.png"); self.make("b.zip")
        moves = organizer.organize_folder(self.root, log=lambda m: None)
        organizer.undo_moves(moves, log=lambda m: None)
        self.assertTrue((self.root / "a.png").exists())
        self.assertTrue((self.root / "b.zip").exists())

    def test_undo_when_original_spot_taken(self):
        self.make("a.png", "first")
        moves = organizer.organize_folder(self.root, log=lambda m: None)
        self.make("a.png", "second")  # کسی همان اسم را دوباره ساخت
        organizer.undo_moves(moves, log=lambda m: None)
        self.assertEqual((self.root / "a.png").read_text(), "second")
        self.assertEqual((self.root / "a (1).png").read_text(), "first")


if __name__ == "__main__":
    unittest.main()
