#ifndef GUI_FUSION_HPP
#define GUI_FUSION_HPP

#include <gtk/gtk.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <mutex>

class MoilUndistorter;

class GuiFusion {
public:
    GuiFusion(MoilUndistorter* moil_undistorter, bool headless, int initial_exposure);
    ~GuiFusion();

    void show_all();
    void update_image(const cv::Mat& frame_bgr);
    int get_key();
    void queue_key(int key_code);

    void set_status_calib(const std::string& text);
    void set_status_ai(const std::string& text);
    void set_setup_hint(const std::string& text);

    void enter_setup_mode(const std::string& calib_name);

    bool is_normalize_enabled() const;
<<<<<<< Updated upstream
=======
    bool is_bw_enabled() const;
>>>>>>> Stashed changes
    bool is_alive() const { return alive_; }
    void wait_for_calibration_ready();
    bool is_calibration_ready() const;

private:
    MoilUndistorter* moil_undistorter_;
    bool headless_;
    int initial_exposure_;
    bool alive_;
    bool normalize_enabled_;
<<<<<<< Updated upstream
=======
    bool bw_enabled_;
>>>>>>> Stashed changes

    GtkWidget* window_;
    GtkWidget* image_;
    GtkWidget* lbl_status_calib_;
    GtkWidget* lbl_status_ai_;
    GtkWidget* entry_exposure_;
    GtkWidget* entry_alpha_;
    GtkWidget* entry_beta_;
    GtkWidget* entry_zoom_;
    GtkWidget* btn_start_calib_;
    GtkWidget* btn_next_step_;
    GtkWidget* chk_normalize_;
<<<<<<< Updated upstream
=======
    GtkWidget* chk_bw_;
>>>>>>> Stashed changes
    GtkWidget* lbl_setup_hint_;

    std::queue<int> key_queue_;
    std::mutex key_mutex_;
    double last_ui_frame_t_;

    bool calibration_ready_;
    std::mutex calib_mutex_;

    void setup_ui();
    static gboolean update_image_idle(gpointer data);
    
    // Callbacks
    static void on_destroy(GtkWidget* widget, gpointer data);
    static gboolean on_key_press(GtkWidget* widget, GdkEventKey* event, gpointer data);

    static void on_btn_exp_clicked(GtkWidget* widget, gpointer data);
    static void on_apply_exp_clicked(GtkWidget* widget, gpointer data);
    
    static void on_btn_alpha_clicked(GtkWidget* widget, gpointer data);
    static void on_btn_beta_clicked(GtkWidget* widget, gpointer data);
    static void on_btn_zoom_clicked(GtkWidget* widget, gpointer data);
    
    static void on_apply_anypoint_clicked(GtkWidget* widget, gpointer data);
    static void on_reset_anypoint_clicked(GtkWidget* widget, gpointer data);
    
    static void on_start_calib_clicked(GtkWidget* widget, gpointer data);
    static void on_next_step_clicked(GtkWidget* widget, gpointer data);
    static void on_chk_normalize_toggled(GtkToggleButton* togglebutton, gpointer data);
<<<<<<< Updated upstream
=======
    static void on_chk_bw_toggled(GtkToggleButton* togglebutton, gpointer data);
>>>>>>> Stashed changes
    
    static void on_action_btn_clicked(GtkWidget* widget, gpointer data);

    void apply_anypoint();
};

#endif // GUI_FUSION_HPP
