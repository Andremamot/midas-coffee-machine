#include "gui_fusion.hpp"
#include "moildev_applicator.hpp"
#include <camera/camera.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <atomic>

/* Shared with run_fusion.cpp — true once gtk_main() event loop is running */
extern std::atomic<bool> g_gui_ready;

struct IdleImageData {
    GuiFusion* gui;
    GByteArray* bytes;
    int w;
    int h;
};

struct AdjData {
    GuiFusion* gui;
    GtkWidget* entry;
    float delta;
};

struct ActionData {
    GuiFusion* gui;
    int key;
};

GuiFusion::GuiFusion(MoildevApplicator* moil_undistorter, bool headless, int initial_exposure, Camera* camera)
    : moil_undistorter_(moil_undistorter),
      headless_(headless),
      initial_exposure_(initial_exposure),
      camera_(camera),
      alive_(true),
      normalize_enabled_(false),
      bw_enabled_(false),
      last_ui_frame_t_(0.0),
      calibration_ready_(false)
{
    if (!headless_) {
        setup_ui();
    }
}

GuiFusion::~GuiFusion() {
}

void GuiFusion::setup_ui() {
    window_ = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window_), "ArUco + MiDaS | Fusion Interface (C++)");
    gtk_window_set_default_size(GTK_WINDOW(window_), 1280, 720);
    g_signal_connect(window_, "destroy", G_CALLBACK(on_destroy), this);
    g_signal_connect(window_, "key-press-event", G_CALLBACK(on_key_press), this);

    GtkWidget* hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    gtk_container_add(GTK_CONTAINER(window_), hbox);

    // Sidebar
    GtkWidget* vbox_ctrl = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(vbox_ctrl), 10);
    gtk_widget_set_size_request(vbox_ctrl, 320, -1);
    gtk_box_pack_start(GTK_BOX(hbox), vbox_ctrl, FALSE, FALSE, 0);

    // System Status
    GtkWidget* f_sys = gtk_frame_new("System Status");
    GtkWidget* vb_s = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(vb_s), 5);
    
    lbl_status_calib_ = gtk_label_new("Mode: Live");
    gtk_widget_set_halign(lbl_status_calib_, GTK_ALIGN_START);
    lbl_status_ai_ = gtk_label_new("AI: Ready");
    gtk_widget_set_halign(lbl_status_ai_, GTK_ALIGN_START);
    
    gtk_box_pack_start(GTK_BOX(vb_s), lbl_status_calib_, FALSE, FALSE, 0);
    gtk_box_pack_start(GTK_BOX(vb_s), lbl_status_ai_, FALSE, FALSE, 0);
    gtk_container_add(GTK_CONTAINER(f_sys), vb_s);
    gtk_box_pack_start(GTK_BOX(vbox_ctrl), f_sys, FALSE, FALSE, 5);

    // Camera Controls
    GtkWidget* f_cam = gtk_frame_new("Camera Controls");
    GtkWidget* vb_c = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(vb_c), 5);

    auto create_adj_row = [&](const char* label, GtkWidget*& entry, float def_val, float delta, GtkWidget* parent_box, void (*callback)(GtkWidget*, gpointer)) {
        GtkWidget* hb = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
        GtkWidget* lbl = gtk_label_new(label);
        gtk_box_pack_start(GTK_BOX(hb), lbl, FALSE, FALSE, 0);
        
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << def_val;
        entry = gtk_entry_new();
        gtk_entry_set_text(GTK_ENTRY(entry), ss.str().c_str());
        gtk_entry_set_width_chars(GTK_ENTRY(entry), 6);

        GtkWidget* btn_min = gtk_button_new_with_label(" - ");
        AdjData* d_min = new AdjData{this, entry, -delta};
        g_signal_connect_data(btn_min, "clicked", G_CALLBACK(callback), d_min, (GClosureNotify) [](gpointer data, GClosure*){ delete (AdjData*)data; }, (GConnectFlags)0);
        
        GtkWidget* btn_plus = gtk_button_new_with_label(" + ");
        AdjData* d_plus = new AdjData{this, entry, delta};
        g_signal_connect_data(btn_plus, "clicked", G_CALLBACK(callback), d_plus, (GClosureNotify) [](gpointer data, GClosure*){ delete (AdjData*)data; }, (GConnectFlags)0);
        
        gtk_box_pack_start(GTK_BOX(hb), btn_min, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(hb), entry, TRUE, TRUE, 0);
        gtk_box_pack_start(GTK_BOX(hb), btn_plus, FALSE, FALSE, 0);
        gtk_box_pack_start(GTK_BOX(parent_box), hb, FALSE, FALSE, 0);
    };

    float display_val = std::max(1.0f, std::min(10.0f, std::round(initial_exposure_ / 1000.0f * 10.0f) / 10.0f));
    if (initial_exposure_ == 0) display_val = 1.0f; // Default if not passed
    create_adj_row("Smart Exp:", entry_exposure_, display_val, 0.5f, vb_c, on_btn_exp_clicked);
    
    GtkWidget* hb_cam_btns = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* btn_apply_exp = gtk_button_new_with_label("Apply Exposure");
    g_signal_connect(btn_apply_exp, "clicked", G_CALLBACK(on_apply_exp_clicked), this);
    gtk_box_pack_start(GTK_BOX(hb_cam_btns), btn_apply_exp, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vb_c), hb_cam_btns, FALSE, FALSE, 5);

    gtk_container_add(GTK_CONTAINER(f_cam), vb_c);
    gtk_box_pack_start(GTK_BOX(vbox_ctrl), f_cam, FALSE, FALSE, 5);

    // Moildev
    GtkWidget* f_moil = gtk_frame_new("Moildev Anypoint Controls");
    GtkWidget* vb_m = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(vb_m), 5);



    float def_pitch = moil_undistorter_ ? moil_undistorter_->pitch_deg()   : 0.0f;
    float def_yaw   = moil_undistorter_ ? moil_undistorter_->yaw_deg()     : 0.0f;
    float def_zoom  = moil_undistorter_ ? moil_undistorter_->zoom_factor() : 1.0f;  // default 1.0 = tidak ada zoom tambahan

    create_adj_row("Alpha (Pitch):", entry_alpha_, def_pitch, 5.0f, vb_m, on_btn_alpha_clicked);
    create_adj_row("Beta (Yaw):  ", entry_beta_, def_yaw, 5.0f, vb_m, on_btn_alpha_clicked);
    create_adj_row("Zoom:        ", entry_zoom_, def_zoom, 0.1f, vb_m, on_btn_alpha_clicked);

    GtkWidget* hb_btns = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget* btn_apply = gtk_button_new_with_label("Apply Anypoint");
    g_signal_connect(btn_apply, "clicked", G_CALLBACK(on_apply_anypoint_clicked), this);
    GtkWidget* btn_reset = gtk_button_new_with_label("Reset View");
    g_signal_connect(btn_reset, "clicked", G_CALLBACK(on_reset_anypoint_clicked), this);
    gtk_box_pack_start(GTK_BOX(hb_btns), btn_apply, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(hb_btns), btn_reset, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vb_m), hb_btns, FALSE, FALSE, 5);

    gtk_container_add(GTK_CONTAINER(f_moil), vb_m);
    gtk_box_pack_start(GTK_BOX(vbox_ctrl), f_moil, FALSE, FALSE, 5);

    // Actions
    GtkWidget* f_act = gtk_frame_new("Actions");
    GtkWidget* vb_a = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    gtk_container_set_border_width(GTK_CONTAINER(vb_a), 5);

    btn_start_calib_ = gtk_button_new_with_label(u8"▶ Start Calibration");
    g_signal_connect(btn_start_calib_, "clicked", G_CALLBACK(on_start_calib_clicked), this);
    gtk_box_pack_start(GTK_BOX(vb_a), btn_start_calib_, FALSE, FALSE, 0);

    btn_next_step_ = gtk_button_new_with_label(u8"⏭ Next Step (Space)");
    g_signal_connect(btn_next_step_, "clicked", G_CALLBACK(on_next_step_clicked), this);
    gtk_box_pack_start(GTK_BOX(vb_a), btn_next_step_, FALSE, FALSE, 0);

    chk_normalize_ = gtk_check_button_new_with_label("Normalize Lighting");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(chk_normalize_), FALSE);
    g_signal_connect(chk_normalize_, "toggled", G_CALLBACK(on_chk_normalize_toggled), this);
    gtk_box_pack_start(GTK_BOX(vb_a), chk_normalize_, FALSE, FALSE, 5);

    chk_bw_ = gtk_check_button_new_with_label("Black & White Mode");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(chk_bw_), FALSE);
    g_signal_connect(chk_bw_, "toggled", G_CALLBACK(on_chk_bw_toggled), this);
    gtk_box_pack_start(GTK_BOX(vb_a), chk_bw_, FALSE, FALSE, 5);

    lbl_setup_hint_ = gtk_label_new("");
    gtk_label_set_line_wrap(GTK_LABEL(lbl_setup_hint_), TRUE);
    gtk_box_pack_start(GTK_BOX(vb_a), lbl_setup_hint_, FALSE, FALSE, 0);

    auto add_action_btn = [&](const char* label, int key) {
        GtkWidget* btn = gtk_button_new_with_label(label);
        ActionData* adata = new ActionData{this, key};
        g_signal_connect_data(btn, "clicked", G_CALLBACK(on_action_btn_clicked), adata, (GClosureNotify) [](gpointer data, GClosure*){ delete (ActionData*)data; }, (GConnectFlags)0);
        gtk_box_pack_start(GTK_BOX(vb_a), btn, FALSE, FALSE, 0);
    };

    add_action_btn("Toggle Record (V)", 'v');
    add_action_btn("Screenshot (S)", 's');
    add_action_btn("Quit (ESC)", 27);

    gtk_container_add(GTK_CONTAINER(f_act), vb_a);
    gtk_box_pack_start(GTK_BOX(vbox_ctrl), f_act, FALSE, FALSE, 5);

    // Image Feed
    image_ = gtk_image_new();
    gtk_box_pack_start(GTK_BOX(hbox), image_, TRUE, TRUE, 0);

    /* NOTE: Do NOT call gtk_widget_show_all() here.
     * On Wayland (Renesas Yocto), the compositor only processes surface
     * creation events AFTER gtk_main() has started its event loop.
     * Calling show_all() here (before gtk_main) causes the window to
     * never be rendered — the Wayland surface commit is silently dropped.
     * show_all() is called from main() just before gtk_main() starts.
     */
}

void GuiFusion::show_all() {
    if (window_) {
        gtk_widget_show_all(window_);
        gtk_widget_hide(btn_start_calib_);
        gtk_widget_hide(btn_next_step_);
        gtk_widget_hide(lbl_setup_hint_);
    }
}

void GuiFusion::update_image(const cv::Mat& frame_bgr) {
    if (headless_ || !alive_) return;
    /* NOTE: g_idle_add() sebelum gtk_main() tetap valid di GTK —
     * callbacks antri di GLib main loop dan dieksekusi saat gtk_main() mulai.
     * Tidak perlu guard g_gui_ready di sini. */

    double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - last_ui_frame_t_ < 0.05) return; // limit to 20 FPS
    last_ui_frame_t_ = now;

    int w = frame_bgr.cols;
    int h = frame_bgr.rows;
    int target_w = 960;
    cv::Mat resized;
    if (w > target_w) {
        float scale = (float)target_w / w;
        cv::resize(frame_bgr, resized, cv::Size(target_w, int(h * scale)), 0, 0, cv::INTER_AREA);
    } else {
        resized = frame_bgr.clone();
    }

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    GByteArray* bytes = g_byte_array_sized_new(rgb.total() * rgb.elemSize());
    g_byte_array_append(bytes, rgb.data, rgb.total() * rgb.elemSize());

    IdleImageData* data = new IdleImageData{this, bytes, rgb.cols, rgb.rows};
    g_idle_add(update_image_idle, data);
}

gboolean GuiFusion::update_image_idle(gpointer user_data) {
    IdleImageData* data = static_cast<IdleImageData*>(user_data);
    if (data->gui->alive_) {
        GBytes* gbytes = g_byte_array_free_to_bytes(data->bytes);
        GdkPixbuf* pixbuf = gdk_pixbuf_new_from_bytes(gbytes, GDK_COLORSPACE_RGB, FALSE, 8, data->w, data->h, data->w * 3);
        gtk_image_set_from_pixbuf(GTK_IMAGE(data->gui->image_), pixbuf);
        g_object_unref(pixbuf);
        g_bytes_unref(gbytes);
    } else {
        g_byte_array_free(data->bytes, TRUE);
    }
    delete data;
    return G_SOURCE_REMOVE;
}

int GuiFusion::get_key() {
    std::lock_guard<std::mutex> lock(key_mutex_);
    if (key_queue_.empty()) return -1;
    int k = key_queue_.front();
    key_queue_.pop();
    return k;
}

void GuiFusion::queue_key(int key_code) {
    std::lock_guard<std::mutex> lock(key_mutex_);
    key_queue_.push(key_code);
}

bool GuiFusion::is_normalize_enabled() const {
    return normalize_enabled_;
}

bool GuiFusion::is_bw_enabled() const {
    return bw_enabled_;
}

void GuiFusion::set_status_calib(const std::string& text) {
    if (headless_ || !g_gui_ready.load()) return;
    g_idle_add([](gpointer data) -> gboolean {
        auto p = static_cast<std::pair<GuiFusion*, std::string>*>(data);
        if (p->first->alive_) gtk_label_set_text(GTK_LABEL(p->first->lbl_status_calib_), p->second.c_str());
        delete p;
        return G_SOURCE_REMOVE;
    }, new std::pair<GuiFusion*, std::string>(this, text));
}

void GuiFusion::set_status_ai(const std::string& text) {
    if (headless_ || !g_gui_ready.load()) return;
    g_idle_add([](gpointer data) -> gboolean {
        auto p = static_cast<std::pair<GuiFusion*, std::string>*>(data);
        if (p->first->alive_) gtk_label_set_text(GTK_LABEL(p->first->lbl_status_ai_), p->second.c_str());
        delete p;
        return G_SOURCE_REMOVE;
    }, new std::pair<GuiFusion*, std::string>(this, text));
}

void GuiFusion::enter_setup_mode(const std::string& calib_name) {
    if (headless_) return;
    /* NOTE: g_idle_add() aman sebelum gtk_main() — callback antri di GLib
     * main loop dan dieksekusi saat gtk_main() mulai.
     * JANGAN pakai guard g_gui_ready di sini! enter_setup_mode() dipanggil
     * dari worker_thread SEBELUM gtk_main() start. */
    std::string hint = "Setup mode: adjust exposure & anypoint,\nthen click \u25B6 Start " + calib_name;
    std::string calib_label = "Mode: Setup - " + calib_name;
    g_idle_add([](gpointer data) -> gboolean {
        auto p = static_cast<std::pair<GuiFusion*, std::pair<std::string, std::string>>*>(data);
        if (p->first->alive_) {
            gtk_widget_show(p->first->btn_start_calib_);
            gtk_widget_show(p->first->lbl_setup_hint_);
            gtk_label_set_text(GTK_LABEL(p->first->lbl_setup_hint_), p->second.first.c_str());
            gtk_label_set_text(GTK_LABEL(p->first->lbl_status_calib_), p->second.second.c_str());
        }
        delete p;
        return G_SOURCE_REMOVE;
    }, new std::pair<GuiFusion*, std::pair<std::string, std::string>>(
        this, {hint, calib_label}));
}

void GuiFusion::wait_for_calibration_ready() {
    while (!calibration_ready_ && alive_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

bool GuiFusion::is_calibration_ready() const {
    return calibration_ready_;
}

// ================= Callbacks =================

void GuiFusion::on_destroy(GtkWidget* widget, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    self->alive_ = false;
    self->queue_key(27);
    gtk_main_quit();
}

gboolean GuiFusion::on_key_press(GtkWidget* widget, GdkEventKey* event, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    if (event->keyval == GDK_KEY_Escape) {
        self->queue_key(27);
    } else if (event->keyval == GDK_KEY_space) {
        self->queue_key(' ');
    } else if (event->keyval < 256) {
        self->queue_key(std::tolower(event->keyval));
    }
    return FALSE;
}

void GuiFusion::on_btn_alpha_clicked(GtkWidget* widget, gpointer data) {
    AdjData* d = static_cast<AdjData*>(data);
    try {
        float val = std::stof(gtk_entry_get_text(GTK_ENTRY(d->entry))) + d->delta;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << val;
        gtk_entry_set_text(GTK_ENTRY(d->entry), ss.str().c_str());
        // Tidak auto-apply map. User harus klik "Apply Anypoint".
    } catch (...) {}
}

void GuiFusion::on_btn_exp_clicked(GtkWidget* widget, gpointer data) {
    AdjData* d = static_cast<AdjData*>(data);
    try {
        float val = std::stof(gtk_entry_get_text(GTK_ENTRY(d->entry))) + d->delta;
        if (val < 1.0f) val = 1.0f;
        if (val > 10.0f) val = 10.0f;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << val;
        gtk_entry_set_text(GTK_ENTRY(d->entry), ss.str().c_str());
        /* Tidak auto-apply — user harus klik "Apply Exposure" */
    } catch (...) {}
}

void GuiFusion::on_apply_exp_clicked(GtkWidget* widget, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    try {
        float val = std::stof(gtk_entry_get_text(GTK_ENTRY(self->entry_exposure_)));
        std::cout << "[GUI] Applying smart exposure: " << val << "\n";
        if (self->camera_) {
            self->camera_->set_smart_exposure(val);
        }
    } catch (...) {}
}

void GuiFusion::on_apply_anypoint_clicked(GtkWidget* widget, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    self->apply_anypoint();
}

void GuiFusion::apply_anypoint() {
    if (!moil_undistorter_) return;
    try {
        float a = std::stof(gtk_entry_get_text(GTK_ENTRY(entry_alpha_)));
        float b = std::stof(gtk_entry_get_text(GTK_ENTRY(entry_beta_)));
        float z = std::stof(gtk_entry_get_text(GTK_ENTRY(entry_zoom_)));
        moil_undistorter_->update_maps(a, b, 0.0f, z);
        // Sinkronkan entry dengan nilai aktual (sama seperti Python)
        // hybrid zoom mungkin membulatkan zoom_ ke nilai presisi yang berbeda
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << moil_undistorter_->zoom_factor();
        gtk_entry_set_text(GTK_ENTRY(entry_zoom_), ss.str().c_str());
    } catch (...) {}
}

void GuiFusion::on_reset_anypoint_clicked(GtkWidget* widget, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    gtk_entry_set_text(GTK_ENTRY(self->entry_alpha_), "0.0");
    gtk_entry_set_text(GTK_ENTRY(self->entry_beta_),  "0.0");
    gtk_entry_set_text(GTK_ENTRY(self->entry_zoom_),  "1.0");
    if (self->moil_undistorter_) {
        self->moil_undistorter_->update_maps(0.0f, 0.0f, 0.0f, 1.0f);
        // Sinkronkan entry dengan nilai aktual (sama seperti Python)
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2) << self->moil_undistorter_->zoom_factor();
        gtk_entry_set_text(GTK_ENTRY(self->entry_zoom_), ss.str().c_str());
    }
}

void GuiFusion::on_start_calib_clicked(GtkWidget* widget, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    self->calibration_ready_ = true;
    gtk_widget_hide(self->btn_start_calib_);
    gtk_widget_show(self->btn_next_step_);
    gtk_label_set_text(GTK_LABEL(self->lbl_setup_hint_), "Calibration in progress...\nClick 'Next Step' when prompted.");
}

void GuiFusion::on_next_step_clicked(GtkWidget* widget, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    self->queue_key(' ');
}

void GuiFusion::on_chk_normalize_toggled(GtkToggleButton* togglebutton, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    self->normalize_enabled_ = gtk_toggle_button_get_active(togglebutton);
}

void GuiFusion::on_chk_bw_toggled(GtkToggleButton* togglebutton, gpointer data) {
    GuiFusion* self = static_cast<GuiFusion*>(data);
    self->bw_enabled_ = gtk_toggle_button_get_active(togglebutton);
}

void GuiFusion::on_action_btn_clicked(GtkWidget* widget, gpointer data) {
    ActionData* d = static_cast<ActionData*>(data);
    d->gui->queue_key(d->key);
}

