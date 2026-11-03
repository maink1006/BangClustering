# BANG Implementation Explanation

## 1. Phạm vi và nguyên tắc

Tài liệu này giải thích phiên bản `ml_bang.py` được viết lại theo hướng:

- tập trung vào **Binary Directory** của BANG-Clustering;
- bám sát mô tả thuật toán trong bài báo Schikuta & Erhart (1997);
- đối chiếu cách tổ chức directory, region, level và API với `bang.py` của pyclustering;
- loại bỏ những phần mở rộng không còn phù hợp với phạm vi;
- phân biệt rõ **thuật toán trong bài báo**, **quy ước triển khai**, và **tiện ích mở rộng**.

Phiên bản này **không** xây dựng lại đầy đủ BANG File theo kiểu cơ sở dữ liệu. Cụ thể, nó không có:

- bucket capacity/overflow;
- data-block 1:m mapping;
- nested-region directory với promoted entries;
- redistribution để cân bằng bucket.

Đây là một giới hạn có chủ ý, không phải tuyên bố đã cài đặt đầy đủ BANG File.

### Thứ tự ưu tiên nguồn

1. **Schikuta & Erhart (1997)** — nguồn lý thuyết chính.
2. **pyclustering `bang.py`** — nguồn đối chiếu triển khai và API.
3. Tài liệu bên ngoài — chỉ dùng khi cần và có ghi chú rõ.

Bài báo định nghĩa BANG-Structure, density index, neighbor search và dendrogram. pyclustering cung cấp cách hiện thực hóa một số quy ước cụ thể, nhưng không phải là nguồn duy nhất để chứng minh tính đúng đắn.

---

## 2. Thuật toán theo bài báo

### 2.1. BANG-Structure và Binary Directory

Theo Section 2.1 của bài báo:

- Không gian dữ liệu k chiều được chia thành các **grid regions** hình hộp chữ nhật.
- Mỗi region được định danh bởi cặp `(region, level)`.
- Region gốc là `(0, 0)`.
- Mỗi lần chia một region sẽ tạo hai region con.
- Các phép chia là nhị phân và chia đôi region theo một chiều.
- Thứ tự các chiều chia phải được xác định thống nhất.
- Các region có thể lồng nhau theo hai axiom:
  - hợp của các region con phủ toàn bộ không gian;
  - nếu hai region giao nhau thì một region chứa region còn lại.

Bài báo cũng nói mỗi data block có thể được ánh xạ bởi nhiều grid region (mapping 1:m), và hợp của chúng là một **block region** hình chữ nhật.

### 2.1.1. Phạm vi của phiên bản này

Phiên bản trong `ml_bang.py` dùng mapping **1:1** giữa leaf grid region và data block:

- mỗi leaf block tương ứng một vùng hình chữ nhật;
- các leaf block tạo thành một phân hoạch rời nhau;
- không có 1:m mapping;
- không có promoted/nested-region directory.

Vì vậy phần lớn nội dung Section 2.1 được giữ ở mức **binary directory**, không phải toàn bộ BANG File.

---

### 2.2. Density Index

Section 2.2 định nghĩa:

```text
D_B = p_B / V_B
```

Trong đó:

- `p_B` là số pattern trong block B;
- `V_B` là tích các extent của block theo từng chiều.

Cài đặt:

```python
BANGBlock._calculate_density()
SpatialBlock.get_volume()
```

Quy ước xử lý trường hợp suy biến:

- Nếu một chiều có extent bằng 0, chiều đó bị bỏ khỏi tích thể tích.
- Nếu tất cả các chiều đều có extent 0, thể tích được đặt bằng 1.0.
- Nếu tích các extent bị underflow, dùng `np.finfo(float).tiny` làm sàn.
- Nếu thể tích overflow hoặc density không hữu hạn, chương trình báo lỗi và yêu cầu chuẩn hóa dữ liệu.

Đây là **quy ước kỹ thuật**, không phải công thức bổ sung của bài báo. Với dữ liệu có đủ các chiều biến thiên, thể tích và density trùng với công thức của bài báo.

Các block lá được sắp xếp theo density tăng dần trong `leaf_blocks`, để tương thích với pyclustering. Khi clustering, danh sách ứng viên được sắp xếp theo density **giảm dần**.

---

### 2.3. Neighbor Search và neighbor degree

Section 2.3 phân biệt:

- **normal neighborhood**: láng giềng theo block regions;
- **refined neighborhood**: láng giềng theo logical regions.

Bài báo định nghĩa **neighbor degree** là số chiều của phần tiếp xúc:

- 0 tương ứng tiếp xúc tại một điểm;
- k-1 tương ứng tiếp xúc qua một siêu mặt k-1 chiều.

Trong 2D:

- chung cạnh: degree 1;
- chung đỉnh: degree 0.

Trong 3D:

- chung mặt: degree 2;
- chung cạnh: degree 1;
- chung đỉnh: degree 0.

Cài đặt:

```python
SpatialBlock.is_neighbor(block)
SpatialBlock.neighbor_degree(block)
```

Cách xác định:

1. Tính độ giao của hai khoảng theo từng chiều.
2. Nếu có chiều bị tách rời: không phải láng giềng.
3. Nếu tất cả các chiều đều giao với độ dài dương: hai hộp chồng lên nhau, không phải tiếp xúc biên.
4. Nếu có ít nhất một chiều chỉ chạm biên: hai block là láng giềng.
5. Neighbor degree bằng số chiều có giao dương.

Cách này xử lý được các block ở **level khác nhau** bằng cách so sánh trực tiếp tọa độ hình học, không cần chuyển đổi region id. Với các region hình chữ nhật, so sánh biên tương đương với việc chuyển region thấp lên level cao để so sánh.

### 2.3.1. Normal và refined neighborhood

Vì phiên bản này dùng 1:1 mapping và leaf regions rời nhau, normal neighborhood và refined neighborhood trùng nhau trên các leaf block. Đây là hệ quả của phạm vi đã chọn, không phải kết luận rằng hai khái niệm giống nhau trong BANG File đầy đủ.

### 2.3.2. Không giới hạn số láng giềng

Không có giới hạn cứng 8 láng giềng. Mọi cặp block thỏa điều kiện tiếp xúc đều được nối.

---

### 2.4. Clustering và dendrogram

Section 2.2 và 2.4 mô tả:

1. Tính density cho các block.
2. Sắp xếp density giảm dần.
3. Bắt đầu từ block có density cao nhất.
4. Tìm các neighbor.
5. Lặp lại neighbor search cho các block đã xử lý.
6. Tạo cluster center mới hoặc kết hợp với cluster hiện có.
7. Ghi các block vào dendrogram theo thứ tự xử lý.

Cài đặt trong `BANGClustering._allocate_clusters()`:

- lấy các leaf block có `density > density_threshold`;
- sắp xếp theo density giảm dần, tie-break bằng `creation_index`;
- dùng BFS với queue;
- mỗi cluster bắt đầu từ block chưa xử lý có density cao nhất;
- khi mở rộng một block, các neighbor chưa xử lý được sắp xếp density giảm dần;
- mỗi block được gán đúng một cluster;
- `dendrogram_` là danh sách cluster, mỗi cluster là danh sách block đã sắp xếp density giảm dần.

Không dùng Union-Find, connected-components library hay một thuật toán phân cụm khác.

### 2.4.1. Hai quy tắc chuỗi trong Section 2.4

Bài báo phát biểu hai trường hợp minh họa:

- Nếu `R1` là neighbor của `R2`, `R2` là neighbor của `R3`, và `R1 > R2 > R3`, thì tạo cluster với `R1, R2, R3`, neighbor search bắt đầu từ `R3`.
- Nếu `R1` là neighbor của `R2`, `R2` là neighbor của `R3`, và `R1 > R2 < R3`, thì tạo cluster với `R1, R2, R3`, neighbor search bắt đầu từ `R2`.

Có thể hiểu định tính:

- Trong chuỗi giảm đơn điệu, `R3` là đầu thấp của chuỗi;
- Trong trường hợp `R2` là cực tiểu cục bộ, `R2` là “valley” nối hai vùng density cao hơn, nên search bắt đầu từ điểm nối này.

Tuy nhiên, bài báo **không cung cấp pseudocode đầy đủ** cho việc áp dụng hai quy tắc này trên toàn bộ đồ thị. pyclustering cũng không hiện thực một cơ chế start-point riêng cho hai trường hợp này.

Phiên bản này:

- bắt đầu mỗi cluster từ block chưa xử lý có density cao nhất, đúng với câu “starting with the first region”;
- xử lý neighbor theo density giảm dần;
- lưu danh sách block của mỗi cluster theo density giảm dần, phù hợp với dendrogram dạng bar trong Figure 8;
- **không tuyên bố** đã tái tạo chính xác từng quyết định start-point cục bộ `R3`/`R2` trong mọi chuỗi.

Đây là một giới hạn cần ghi nhận. Việc tự đặt ra một thuật toán start-point toàn cục rồi gán cho Schikuta & Erhart sẽ là sáng tạo không có căn cứ.

---

## 3. Kiến trúc code

### 3.1. `SpatialBlock`

Vai trò: hình học của một vùng.

Thuộc tính:

- `max_corner`, `min_corner`;
- `upper`, `lower` (alias nội bộ);
- `volume_floor`;
- `_volume`.

Phương thức:

- `get_corners()`;
- `get_volume()`;
- `contains(points)`;
- `split(dimension)`: chia tại trung điểm;
- `is_neighbor(block)`;
- `neighbor_degree(block)`.

Constructor theo thứ tự pyclustering:

```python
SpatialBlock(max_corner, min_corner)
```

### 3.2. `BANGBlock`

Vai trò: một block trong directory.

Thuộc tính:

- `region`, `level`;
- `spatial_block`;
- `point_indices` (NumPy array nội bộ);
- `cluster`;
- `children`, `parent`;
- `creation_index` (dùng tie-break, không phải region id);
- `density`;
- `population` (alias của `size`).

Phương thức tương thích pyclustering:

- `get_region()`;
- `get_density()`;
- `get_cluster()`;
- `get_spatial_block()`;
- `get_points()`;
- `set_cluster(index)`;
- `is_neighbor(block)`;
- `split(dimension)`.

Phương thức mở rộng:

- `points()`;
- `neighbor_degree(block)`.

### 3.3. `BANGDirectory`

Vai trò: xây dựng cây nhị phân, quản lý leaf và level.

Constructor:

```python
BANGDirectory(data, levels, **kwargs)
```

Các keyword:

- `observe=True`;
- `density_threshold=0.0`;
- `amount_threshold=0`.

Phương thức:

- `get_data()`;
- `get_leafs()`;
- `get_level(level)`;
- `get_height()`;
- `get_root()`;
- `iter_all_blocks()`;
- `__len__()`.

Quy ước level:

- root ở level 0;
- `levels` là số level tối đa, tính cả root;
- tối đa `levels - 1` lần chia trên một đường từ root đến leaf.

### 3.4. `BANGClustering`

API fit-style, là phần mở rộng so với pyclustering:

```python
model = BANGClustering(
    levels=8,
    density_threshold=0.0,
    amount_threshold=0,
    observe=True,
)
model.fit(X)
```

Kết quả:

- `labels_`: NumPy array, `-1` là noise;
- `clusters_`: list các NumPy array;
- `noise_`: NumPy array;
- `cluster_blocks_`: list các cluster, mỗi cluster là list block;
- `dendrogram_`: list các cluster block, đã sắp xếp density giảm dần;
- `processing_trace_`: lịch sử xử lý block;
- `audit_report_`: báo cáo invariant.

### 3.5. `bang`

Facade tương thích pyclustering:

```python
algorithm = bang(data, levels, ccore=False, **kwargs)
algorithm.process()
clusters = algorithm.get_clusters()
noise = algorithm.get_noise()
directory = algorithm.get_directory()
dendrogram = algorithm.get_dendrogram()
encoding = algorithm.get_cluster_encoding()
```

`ccore` được giữ trong chữ ký để tương thích nhưng không được sử dụng.

### 3.6. Visualizer và Animator

- `BANGVisualizer`: API mở rộng có constructor, dùng cho notebook.
- `bang_visualizer`: facade static tương thích pyclustering.
- `bang_animator`: animation 2D/projection.
- `BANGAnimator`: alias mở rộng.

---

## 4. Luồng xử lý

### 4.1. Xây dựng directory

1. Kiểm tra dữ liệu là ma trận 2D, hữu hạn, không rỗng.
2. Lấy max/min toàn cục.
3. Tạo root `(0,0)`.
4. Với level từ 1 đến `levels - 1`:
   - duyệt các block ở level hiện tại;
   - nếu block rỗng, quá ít điểm, hoặc density thấp: thành leaf;
   - nếu không: chia theo chiều `level % k`;
   - tạo hai block con;
   - con trái giữ region của cha;
   - con phải có region `parent.region + 2 ** parent.level`;
   - level con bằng `parent.level + 1`.
5. Gắn leaf vào danh sách, sắp xếp density tăng dần.
6. Kiểm tra leaf tạo thành phân hoạch đầy đủ, rời nhau.

### 4.2. Chia điểm

`BANGBlock.split(dimension)`:

1. Lấy border từ `SpatialBlock.split`.
2. Điểm có tọa độ `< border` thuộc con trái.
3. Điểm còn lại thuộc con phải.

Quy ước này bảo đảm điểm nằm đúng trên biên chỉ thuộc một leaf. Đây là **quy ước triển khai** để bảo toàn invariant phân hoạch; pyclustering dùng block.contains inclusive và có thể trả một điểm ở cả hai phía khi kiểm tra trực tiếp.

### 4.3. Clustering

1. Lấy các leaf có density > `density_threshold`.
2. Sắp xếp density giảm dần.
3. Chọn seed đầu tiên chưa xử lý.
4. BFS:
   - gán cluster;
   - lấy các neighbor chưa xử lý;
   - sắp xếp neighbor density giảm dần;
   - đưa vào queue.
5. Lặp đến hết candidate.
6. Sắp xếp lại danh sách block của mỗi cluster theo density giảm dần.
7. Gán nhãn điểm; block không thuộc cluster là noise.

### 4.4. Audit

`get_audit_report()` kiểm tra:

- partition leaf đầy đủ và rời;
- region/level của con đúng công thức;
- mỗi cặp `(region, level)` là duy nhất;
- density bằng population/volume;
- cluster block đúng candidate set, không trùng;
- mỗi cluster block sorted density giảm dần;
- `dendrogram_` khớp danh sách cluster block;
- nhãn cluster khớp `clusters_` và `noise_`;
- processing trace phủ toàn bộ candidate.

Nếu audit thất bại, `fit()` raise `RuntimeError`.

---

## 5. Công thức và quy ước

### 5.1. Region id

Theo quy ước pyclustering:

```text
left.region  = parent.region
right.region = parent.region + 2 ** parent.level
left.level   = right.level = parent.level + 1
```

Region id **không phải** heap id `2*r+1`, `2*r+2`. `creation_index` chỉ là số thứ tự nội bộ để tie-break, không được dùng làm region.

### 5.2. Chiều chia

Bài báo yêu cầu một thứ tự chia chiều được xác định thống nhất nhưng không nêu chiều bắt đầu. Phiên bản này dùng quy ước của pyclustering:

```python
split_dimension = new_level % k
```

Đây là **quy ước triển khai tương thích pyclustering**, không phải công thức được bài báo phát biểu. Nếu đổi chiều bắt đầu, cần ghi rõ là biến thể.

### 5.3. Volume và density

Input thường:

```text
V_B = ∏_{i=1..k} (max_i - min_i)
D_B = p_B / V_B
```

Cài đặt bỏ các chiều có extent 0 và dùng:

```text
V_B = 1.0                    nếu mọi extent đều 0
V_B = max(∏ positive sides, tiny)  nếu có ít nhất một extent dương
```

Đây là quy ước kỹ thuật cho dữ liệu suy biến/underflow. Với dữ liệu thông thường, công thức trùng bài báo.

### 5.4. Neighbor degree

Với hai hộp chạm nhau:

```text
degree = số chiều mà hai khoảng giao nhau với độ dài > tolerance
```

Ví dụ 3D:

- chung mặt: 2;
- chung cạnh: 1;
- chung đỉnh: 0.

`is_neighbor()` trả `True` nếu chúng không bị tách rời và có ít nhất một chiều chỉ chạm biên. Hai hộp chồng lấn hoàn toàn không được coi là láng giềng biên trong phiên bản 1:1.

---

## 6. Những thay đổi so với `ml_bang.py` cũ

| Thành phần | Cách triển khai cũ | Vấn đề | Cách sửa mới | Cơ sở lý thuyết |
|---|---|---|---|---|
| Region id | Heap id `2*r+1`, `2*r+2` | Không phải quy ước BANG/pyclustering; lẫn `region`, `level`, node id | Dùng công thức left = parent.region, right = parent.region + `2**parent.level` | Section 2.1; đối chiếu pyclustering |
| `level` của con | Tăng nhưng dùng heap id | Không nhất quán với region | Level con = parent.level + 1; root level 0 | Section 2.1, pyclustering |
| Chiều chia | `(level - 1) % k` | Khác pyclustering | `new_level % k` | Quy ước pyclustering; bài báo chỉ yêu cầu thứ tự cố định |
| Clustering | Union-Find + tự thiết kế merge tree | Không phải quy trình density-order/neighbor search của Section 2.4; dendrogram không phải block-list | BFS density-order, neighbor density-order; dendrogram là list block | Section 2.2/2.4 |
| Dendrogram | Dict `nodes`, `roots`, `floor`, `events` gọi là cây | Không tương ứng dendrogram dạng danh sách block trong bài/pyclustering | `get_dendrogram()` trả list cluster block; lịch sử xử lý chuyển thành `get_processing_trace()` | Section 2.4; API pyclustering |
| `partition_strategy`, `bucket_capacity`, `spread`, `median` | Có trong code cũ | Ngoài phạm vi Binary Directory; không có đủ cơ sở trong bài cho batch split | Loại bỏ | Phạm vi yêu cầu |
| `connectivity`, `max_contact_dims` | Lọc loại tiếp xúc theo tên tùy ý | Không phải neighbor degree của bài báo; dễ đổi nghĩa face/edge | Loại bỏ; dùng `neighbor_degree` và `is_neighbor` theo hình học | Section 2.3 |
| Số láng giềng | Có giới hạn/ xử lý BFS tùy biến | Có thể bỏ sót neighbor | Duyệt đầy đủ mọi cặp block; không cap | Section 2.3; pyclustering cap 8 là hạn chế triển khai |
| `amount_threshold` | Dùng làm filter candidate và threshold | Không có trong bài; khác pyclustering | Chỉ dùng để dừng tách directory; density threshold quyết định noise | Extension pyclustering; ghi rõ |
| Boundary point | Dễ trùng ở hai block do contains inclusive | Vi phạm một điểm một leaf | Chia index theo `< border`; leaf partition check | Invariant triển khai |
| Constant feature / zero volume | Có thể chia cho 0 | Density không xác định | Bỏ extent 0; toàn zero → volume 1; kiểm tra số học | Quy ước kỹ thuật |
| `observe` | Không có | Luôn lưu level hoặc không lưu? | Thêm `observe`; lưu list hoặc lazy traverse | API pyclustering + yêu cầu |
| API pyclustering | Thiếu wrapper | Khó dùng thay thế | Thêm `bang`, alias directory/block/spatial, visualizer/animator | Source 2 |
| Visualizer | Có plot tùy biến, dendrogram dict | Không tương thích show_dendrogram bloc-list | `bang_visualizer` static; `BANGVisualizer` extension | API pyclustering |
| Animation | Không có | Thiếu yêu cầu | `bang_animator` với các stage và save | Section 3 + API pyclustering |
| Silhouette/evaluation | Có trong file cũ | Không thuộc BANG algorithm | Không đưa vào file này | Giữ phạm vi |

---

## 7. Những điểm tương thích với pyclustering

### 7.1. Tương thích tốt

- `bang(data, levels, ccore=False, **kwargs)` và `process()`.
- `get_clusters()`, `get_noise()`, `get_directory()`, `get_dendrogram()`, `get_cluster_encoding()`.
- `bang_directory`/`BANGDirectory` với `get_data()`, `get_leafs()`, `get_level()`, `get_height()`.
- `bang_block`/`BANGBlock` với các getter/setter/is_neighbor.
- `spatial_block` constructor `(max_corner, min_corner)`, `get_corners()`, `get_volume()`, `split()`, `is_neighbor()`.
- Region numbering và split dimension: khớp pyclustering trong 270/270 trường hợp kiểm thử.
- Leaf structure (region, level, density, population): 270/270 trùng.
- Clustering với `amount_threshold=0`: 270/270 trùng cluster/noise.

### 7.2. Khác biệt có chủ ý

1. **Level/height**
   - pyclustering không lưu root trong `__level_blocks`; `get_level(0)` trả level đầu tiên được tạo, và `get_height()` là số level đã lưu.
   - Phiên bản này lưu root ở level 0; `get_level(0)` trả root; `get_height()` tính cả root.
   - Lý do: nhất quán với region gốc `(0,0)` của bài báo.

2. **Dendrogram order**
   - pyclustering ghép block theo thứ tự neighbor được tìm thấy, không đảm bảo list con sorted.
   - Phiên bản này sắp xếp mỗi cluster theo density giảm dần, theo yêu cầu của Section 2.4 và yêu cầu người dùng.

3. **Neighbor test**
   - pyclustering dùng hiệu max-corner và có thể coi hai hộp bị tách rời là neighbor trong một số cấu hình.
   - Phiên bản này kiểm tra giao khoảng theo từng chiều; chặt hơn và đúng điều kiện tiếp xúc biên.

4. **Không cap 8 neighbors**
   - pyclustering dừng neighbor search sau 8 neighbor.
   - Phiên bản này không giới hạn.

5. **`amount_threshold`**
   - pyclustering dùng trong allocation và có thể dừng sớm, để lại noise.
   - Phiên bản này chỉ dùng khi xây directory; noise do `density_threshold`.
   - Với `amount_threshold > 0`, kết quả có thể khác pyclustering.

6. **`get_data()` / `get_clusters()` / `get_noise()`**
   - Trả về list (và list of lists) để gần kiểu pyclustering.
   - `labels_`, `clusters_`, `noise_` là extension dạng NumPy.

7. **`get_cluster_encoding()`**
   - Trả `ml_bang.ClusterEncoding.CLUSTER_INDEX_LIST_SEPARATION` với value 1, tương đương `type_encoding` của pyclustering nhưng không phải cùng enum class.

8. **`observe=False`**
   - pyclustering không hỗ trợ lazy level truy xuất đầy đủ; phiên bản này bổ sung `get_level()` bằng traversal.

9. **Unknown keyword arguments**
   - pyclustering bỏ qua các keyword không dùng.
   - Phiên bản này để `BANGClustering` báo `TypeError` nếu keyword không hợp lệ, nhằm tránh cấu hình bị bỏ sót âm thầm. `ccore` vẫn được nhận và bỏ qua để tương thích chữ ký.

---

## 8. Animation

Bài báo Section 3 mô tả hệ thống BANG-Clustering tương tác, cho phép người dùng theo dõi từng bước của quá trình phân cụm và chọn chiều chiếu cho dữ liệu nhiều chiều. Animation trong `ml_bang.py` lấy ý tưởng giao diện theo dõi từng bước đó; nó là visualization, không phải một thuật toán phân cụm mới.

`bang_animator(directory, clusters, noise=None, feature_indices=(0, 1))`.

Signature `animate(animation_velocity=75, movie_fps=25, movie_filename=None)` giữ tương thích. Các tham số mở rộng:

- `noise`: nếu không truyền, tự suy ra từ các index không thuộc cluster;
- `feature_indices`: chiếu dữ liệu nhiều chiều;
- `processing_trace` (keyword bổ sung ở constructor): nếu truyền trace từ `BANGClustering`, animation theo đúng thứ tự block đã xử lý; nếu không, dùng thứ tự density giảm dần.

Các giai đoạn:

1. Dữ liệu ban đầu.
2. Các level của directory (root → leaf).
3. Leaf density.
4. Xử lý từng block theo thứ tự density (hoặc trace).
5. Kết quả cuối: cluster và noise.

Lưu file:

- `.gif` dùng writer `pillow`;
- `.mp4` dùng writer `ffmpeg`;
- nếu writer không có, raise `RuntimeError` rõ ràng.

Animation chỉ là visualization, không thay đổi clustering.

---

## 9. `observe` và bộ nhớ

### `observe=True`

- Lưu `level_blocks`: list các list block.
- `get_level(level)` O(1).
- Tốn thêm bộ nhớ cho các tham chiếu block theo level.
- Không làm mất block trung gian.

### `observe=False`

- Không lưu `level_blocks`.
- `get_level(level)` duyệt cây từ root, O(số block đến level đó).
- `get_height()` vẫn chính xác.
- Cây vẫn giữ `children`, `parent` và mọi block trung gian; vì vậy **không tiết kiệm toàn bộ bộ nhớ block**.
- Chỉ tiết kiệm bộ nhớ danh sách level, đổi lại truy xuất chậm hơn.

Đây là đánh đổi được ghi rõ, không tuyên bố tiết kiệm toàn bộ.

---

## 10. Kiểm thử và bằng chứng

### 10.1. Self-test trong file

```bash
python3 ml_bang.py
```

Kết quả:

```text
BANG self-tests: 152 checks passed
```

Bao gồm:

- 1D, 2D, 3D;
- feature hằng;
- điểm trùng;
- điểm trên biên;
- block rỗng;
- density bằng 0;
- toàn bộ dữ liệu một block;
- density bằng nhau và tie-break deterministic;
- một cluster / nhiều cluster / toàn noise;
- region formula;
- observation/lazy level;
- neighbor degree face/edge/corner và khác level;
- API facade;
- animation stages cơ bản.

### 10.2. Kiểm thử bổ sung

Đã chạy thêm các test ngoài file:

- partition đầy đủ cho nhiều cấu hình;
- density đúng;
- neighbor sets;
- lưu GIF animation;
- kiểm tra lỗi input.

### 10.3. Đối chiếu pyclustering

Kịch bản đối chiếu: 15 dataset ngẫu nhiên, 1–3 chiều, `levels` 1–6, `density_threshold` 0/0.005/0.02, `amount_threshold=0`. Môi trường đối chiếu dùng `pyclustering 0.10.1.2`.

Kết quả:

- **270/270** leaf structure trùng `(region, level, density, population)`.
- **270/270** cluster/noise trùng.
- **12** khác biệt neighbor set trong kiểm tra bổ sung; pyclustering coi một số hộp bị tách rời là neighbor do kiểm tra max-corner, trong khi phiên bản này kiểm tra giao khoảng theo chiều và loại chúng.
- Với `amount_threshold>0`, 23/160 cấu hình khác biệt; nguyên nhân là pyclustering dùng `amount_threshold` trong allocation và có thể dừng sớm, còn phiên bản này chỉ dùng nó để dừng directory.

Các khác biệt trên được phân loại là:
- sửa lỗi hình học neighbor;
- bỏ giới hạn 8 neighbor;
- semantics `amount_threshold` có chủ ý;
- khác biệt thứ tự/dendrogram có chủ ý.

---

## 11. Hạn chế của phiên bản

1. Không cài đặt đầy đủ BANG File: bucket capacity, overflow, redistribution, 1:m data-block mapping, promoted nested-region directory.
2. Normal/refined neighborhood trùng nhau vì mapping leaf 1:1.
3. Không chứng minh tương đương tuyệt đối với hai quy tắc start-point `R1>R2>R3` và `R1>R2<R3` của Section 2.4; đây là phần bài báo không mô tả đủ để viết pseudocode duy nhất.
4. `amount_threshold` là extension; semantics khác pyclustering khi >0.
5. `get_level(0)` / `get_height()` khác pyclustering vì root được tính.
6. Dendrogram là list block sorted density, không phải discovery-order của pyclustering; cũng không phải cây merge tự thiết kế.
7. Neighbor comparison hiện tại dùng vòng lặp O(L²); phù hợp dữ liệu vừa, chưa tối ưu cho hàng chục nghìn block.
8. Không có out-of-sample prediction; BANG là thuật toán transductive.
9. Các chiều hằng số/volume 0 dùng quy ước kỹ thuật, không phải công thức bài báo.
10. Không đưa các metric đánh giá như silhouette vào file core.

---

## 12. Ví dụ sử dụng

### Fit-style

```python
import numpy as np
from ml_bang import BANGClustering

X = np.random.default_rng(42).normal(size=(500, 2))

model = BANGClustering(
    levels=7,
    density_threshold=0.0,
    amount_threshold=0,
    observe=True,
)
labels = model.fit(X).labels_

print(model.get_clusters())
print(model.get_noise())
print(model.get_dendrogram())
print(model.get_audit_report())
```

### pyclustering style

```python
from ml_bang import bang

algorithm = bang(X.tolist(), levels=7)
algorithm.process()

clusters = algorithm.get_clusters()
noise = algorithm.get_noise()
directory = algorithm.get_directory()
dendrogram = algorithm.get_dendrogram()
```

### Visualization

```python
from ml_bang import bang_visualizer

bang_visualizer.show_blocks(directory)
bang_visualizer.show_clusters(X.tolist(), clusters, noise)
bang_visualizer.show_dendrogram(dendrogram)
```

### Animation

```python
from ml_bang import bang_animator

animator = bang_animator(directory, clusters, noise=noise)
animator.animate(animation_velocity=75)
# hoặc lưu GIF
animator.animate(movie_filename="bang.gif", movie_fps=15)
```

---

## 13. Tài liệu tham khảo

1. Schikuta, E., & Erhart, M. (1997). *The BANG-clustering system: Grid-based data analysis.* In Advances in Intelligent Data Analysis: Reasoning about Data (IDA'97), Lecture Notes in Computer Science, Vol. 1280, pp. 513–524. Springer. DOI: https://doi.org/10.1007/BFb0052867
2. Pyclustering, `pyclustering/cluster/bang.py`. GitHub: https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/bang.py
3. Freeston, M. W. (1987). *The BANG file: a new kind of grid file.* Proceedings of ACM SIGMOD. Được bài báo Schikuta & Erhart trích dẫn cho nền tảng BANG file; tài liệu gốc không được dùng làm nguồn chính trong phiên bản này.
