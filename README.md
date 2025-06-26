# DeepGaitV2

**DeepGaitV2** là hệ thống nhận dạng và phân tích dáng đi (gait recognition) dựa trên mô hình học sâu tiên tiến. Dự án hướng tới việc cải thiện độ chính xác và khả năng tổng quát của các hệ thống nhận dạng dáng đi trong môi trường thực tế phức tạp.

---

## 📌 Mục tiêu

- Phát triển một mô hình học sâu tối ưu để nhận dạng người thông qua dáng đi.
- Hỗ trợ phân tích dáng đi trong điều kiện ánh sáng thay đổi, nhiều góc nhìn và nhiễu nền.
- Ứng dụng vào giám sát an ninh, y học, và tương tác người-máy.

---

## 🧠 Tính năng chính

- ✅ Mô hình học sâu (CNN + Transformer) tối ưu cho nhận dạng chuỗi ảnh dáng đi.
- ✅ Hỗ trợ nhiều định dạng đầu vào: ảnh động (Gait Energy Image), video, hoặc skeleton 2D/3D.
- ✅ Khả năng huấn luyện và đánh giá trên nhiều tập dữ liệu (CASIA-B, OU-MVLP, GREW...).
- ✅ Khả năng nhận diện từ nhiều góc nhìn (cross-view gait recognition).

---

  ## Bài báo: [Exploring Deep Models for Practical Gait Recognition](https://arxiv.org/abs/2303.03301)  
  ## Repo: [OpenGait](https://github.com/ShiqiYu/OpenGait)     
# 2. Cách chạy
  ``` data_multi_view ``` là dataset với 4 góc nhìn 000, 036, 090 và 144 với 3 ID  
  ``` data_single_view ``` là dataset với 1 góc nhìn 090 với 8 ID   
Đầu tiên, chỉnh sửa về file config tại đường dẫn: configs/deepgaitv2/DeepGaitV2_casiab.yaml  
ở mục ``` dataset_root: your_path ```, điều chỉnh thành đường dẫn của mình.  
Điều chỉnh ``` dataset_partition: ./datasets/CASIA-B/CASIA-B.json ```    
để tùy chỉnh TRAIN_SET và TEST_SET.  
Tải model đã train sẵn: [CASIA-B](https://drive.google.com/file/d/1e_ZPE-Igip-i1OUIyFczmx5ChuQjdDhv/view?usp=sharing), [Gait3D](https://drive.google.com/file/d/1uIbOaiZhjgD9TUcsn68uIOIxIUpFbcsA/view?usp=sharing)  
Sau khi tải model xong, copy lại vào đường dẫn tương ứng ``` OpenGait/output/CASIA-B/DeepGaitV2/DeepGaitV2/checkpoints ```   
Sau khi hoàn thành các bước thiết lập, tiến hành chạy thử ở file ``` PBL4.ipynb ```  
## Colab Notebook  
Trong file ``` PBL4.ipynb ``` đã có sẵn các dòng lệnh để:  
- Xử lý dữ liệu: tách frame, segmentation, resize,...  
- Cài đặt môi trường cho dự án  
