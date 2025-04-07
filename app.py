# Import các thư viện cần thiết
import os  # Thư viện để tương tác với hệ thống tập tin
import time  # Thư viện để làm việc với thời gian
import torch  # Framework deep learning PyTorch
import torchvision.transforms as transforms  # Công cụ để xử lý và biến đổi hình ảnh
from PIL import Image  # Thư viện để mở và xử lý hình ảnh
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify  # Framework web
import torch.nn as nn  # Module neural network của PyTorch
from torchvision import models  # Các mô hình deep learning đã được huấn luyện
from werkzeug.utils import secure_filename  # Hàm để đảm bảo tên tập tin an toàn

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
app.secret_key = 'quality_fruit_app_secret_key'  # Khoá bí mật cho phiên làm việc Flask

# Cấu hình thư mục lưu trữ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Lấy đường dẫn thư mục gốc của ứng dụng
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')  # Tạo đường dẫn đến thư mục uploads
QUALITY_MODEL_PATH = r"D:\2025\fruit-classification-project\trained_models\fruit_quality_model.pth"  # Đường dẫn đến file mô hình

# Đảm bảo thư mục upload tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Tạo thư mục uploads nếu chưa tồn tại
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Cấu hình thư mục upload cho Flask

# Các nhãn phân loại chất lượng
quality_classes = ['Bad Quality', 'Good Quality', 'Mixed Quality']  # Các lớp chất lượng trái cây

# Hàm tải mô hình phân loại chất lượng
def load_quality_model():
    model = models.mobilenet_v2(weights=None)  # Tạo mô hình MobileNetV2 không có trọng số
    model.classifier[-1] = nn.Linear(model.last_channel, len(quality_classes))  # Điều chỉnh lớp cuối cho số lượng lớp phân loại
    
    if os.path.exists(QUALITY_MODEL_PATH):  # Kiểm tra xem file mô hình có tồn tại không
        model.load_state_dict(torch.load(QUALITY_MODEL_PATH, map_location=torch.device('cpu')))  # Tải trọng số từ file
        model.eval()  # Chuyển mô hình sang chế độ đánh giá
        return model
    else:
        print(f"Warning: Quality model not found at {QUALITY_MODEL_PATH}")  # In cảnh báo nếu không tìm thấy mô hình
        return None

# Khởi tạo mô hình
quality_model = load_quality_model()  # Tải mô hình phân loại chất lượng

# Hàm dự đoán chất lượng
def predict_quality(image_path):
    if quality_model is None:  # Kiểm tra nếu mô hình chưa được tải
        return "Model not loaded", 0
    
    # Định nghĩa các bước tiền xử lý hình ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh thành 224x224
        transforms.ToTensor(),  # Chuyển đổi ảnh thành tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn hóa ảnh
    ])
    
    image = Image.open(image_path).convert('RGB')  # Mở hình ảnh và chuyển đổi sang RGB
    image_tensor = transform(image).unsqueeze(0)  # Áp dụng biến đổi và thêm chiều batch
    
    with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm bộ nhớ
        output = quality_model(image_tensor)  # Chạy mô hình để có kết quả dự đoán
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Chuyển đổi kết quả thành xác suất
        confidence, predicted = torch.max(probabilities, 1)  # Lấy chỉ số lớp có xác suất cao nhất
        class_name = quality_classes[predicted.item()]  # Lấy tên lớp từ chỉ số
        accuracy = confidence.item() * 100  # Chuyển độ tin cậy thành phần trăm
    
    return class_name, accuracy  # Trả về tên lớp và độ chính xác

# Route chính
@app.route('/')
def index():
    # Kiểm tra xem mô hình đã được tải thành công chưa
    models_status = {
        'quality_model': quality_model is not None
    }
    return render_template('index.html', models_status=models_status)  # Hiển thị trang chủ với trạng thái mô hình

# Route xử lý tải lên và phân loại (không sử dụng nữa, chuyển sang API)
@app.route('/upload', methods=['POST'])
def upload():
    # Chuyển hướng toàn bộ sang API classify_only
    return classify_only()

# Route phân loại (API trả về JSON) - dùng cho cả tab tải lên và tab chụp ảnh
@app.route('/classify_only', methods=['POST'])
def classify_only():
    if 'file' not in request.files:  # Kiểm tra xem có file trong yêu cầu không
        return jsonify({'error': 'Không tìm thấy tệp'}), 400  # Trả về lỗi dạng JSON
    
    file = request.files['file']  # Lấy file từ yêu cầu
    if file.filename == '':  # Kiểm tra xem đã chọn file chưa
        return jsonify({'error': 'Chưa chọn tệp'}), 400  # Trả về lỗi dạng JSON
    
    if file:  # Nếu có file
        # Đảm bảo tên file là duy nhất
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Tạo đường dẫn đầy đủ
        file.save(filepath)  # Lưu file
        
        # Dự đoán chất lượng nếu mô hình đã sẵn sàng
        try:
            result, accuracy = predict_quality(filepath)  # Dự đoán chất lượng
            # Trả về kết quả dạng JSON
            return jsonify({
                'filename': filename,
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'result': result,
                'accuracy': accuracy
            })
        except Exception as e:
            return jsonify({'error': f"Lỗi phân loại: {str(e)}"}), 500  # Trả về lỗi dạng JSON

# Điểm vào ứng dụng
if __name__ == '__main__':
    app.run(debug=True)  # Chạy ứng dụng Flask với chế độ debug bật