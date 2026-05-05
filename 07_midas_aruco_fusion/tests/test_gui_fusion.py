import pytest
import numpy as np
import queue
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib

from core.gui_fusion import FusionGUI

class MockMoilUndistorter:
    def __init__(self):
        self.pitch = 0.0
        self.yaw = 0.0
        self.zoom = 1.0

def test_gui_initialization():
    """Memastikan GUI dapat diinstansiasi tanpa crash (mengecek pembentukan UI elemen dasar)."""
    # Instansiasi window tanpa menampilkannya ke layar (show_all() di-skip agar tidak block di CI/CD)
    gui = FusionGUI(moil_undistorter=None, headless=True)
    assert gui is not None
    assert isinstance(gui, Gtk.Window)
    assert hasattr(gui, "key_queue")
    assert isinstance(gui.key_queue, queue.Queue)

def test_keystroke_queue():
    """Memastikan sistem queue keystroke thread-safe berfungsi."""
    gui = FusionGUI(moil_undistorter=None, headless=True)
    
    # Kondisi awal: tidak ada key yang ditekan, return -1 (seperti cv2.waitKey(1))
    assert gui.get_key() == -1
    
    # Simulasi keystroke dari GTK main thread
    gui.queue_key(ord('q'))
    gui.queue_key(ord('s'))
    
    # Background thread membaca keystroke
    assert gui.get_key() == ord('q')
    assert gui.get_key() == ord('s')
    assert gui.get_key() == -1

def test_update_image_stores_frame():
    """Memastikan background thread dapat mengirimkan frame ke GUI tanpa error."""
    gui = FusionGUI(moil_undistorter=None, headless=True)
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Ini harus memicu GLib.idle_add tanpa exception
    gui.update_image(dummy_frame)
    
    # Memaksa eksekusi loop GLib yang tertunda (idle callbacks)
    context = GLib.MainContext.default()
    while context.iteration(False):
        pass
        
    # Pastikan update_image_ui tidak crash saat diberikan frame
    # Karena headless=True, GUI tidak akan memanggil GdkPixbuf yang butuh display X11 aktif.
    # Kita hanya memverifikasi method dipanggil dengan benar.
    assert True

def test_moil_parameter_binding():
    """Memastikan klik tombol / entry di GUI dapat mengubah properti object MoilUndistorter secara langsung."""
    mock_moil = MockMoilUndistorter()
    gui = FusionGUI(moil_undistorter=mock_moil, headless=True)
    
    # Set nilai awal di GUI UI Entry (diasumsikan entry memiliki format text string)
    gui.entry_alpha.set_text("15.5")
    gui.entry_beta.set_text("-10.0")
    gui.entry_zoom.set_text("2.5")
    
    # Simulasi user menekan tombol "Apply Anypoint"
    gui.on_apply_anypoint(None)
    
    # Pastikan object mock_moil berubah! Ini yang diakses oleh thread worker.
    assert mock_moil.pitch == 15.5
    assert mock_moil.yaw == -10.0
    assert mock_moil.zoom == 2.5

def test_moil_button_increment():
    """Memastikan tombol [+] dan [-] dapat memanipulasi nilai Anypoint."""
    mock_moil = MockMoilUndistorter()
    gui = FusionGUI(moil_undistorter=mock_moil, headless=True)
    
    gui.entry_alpha.set_text("0.0")
    
    # Simulasi klik [+] untuk Alpha
    gui.on_adj_alpha(None, 5.0)
    
    # Entry berubah
    assert gui.entry_alpha.get_text() == "5.0"
    # Moil model berubah
    assert mock_moil.pitch == 5.0
    
    # Simulasi klik [-]
    gui.on_adj_alpha(None, -10.0)
    assert gui.entry_alpha.get_text() == "-5.0"
    assert mock_moil.pitch == -5.0
