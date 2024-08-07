import torch as th
import kornia
import adarl.utils.dbg.ggLog as ggLog

def build_rotmat(a : th.Tensor):
    rotmat = a.new_zeros(size=(3,3))
    ca = th.cos(a)
    sa = th.sin(a)
    rotmat[0,0] = ca
    rotmat[0,1] = -sa
    rotmat[1,0] = sa
    rotmat[1,1] = ca
    rotmat[2,2] = 1
    return rotmat

def build_traslmat(xy : th.Tensor):
    tmat = xy.new_zeros(size=(3,3))
    tmat[0,0] = 1
    tmat[1,1] = 1
    tmat[2,2] = 1
    tmat[0,2] = xy[0]
    tmat[1,2] = xy[1]
    return tmat

def transform_points(points_Nxy : th.Tensor, transform_xya : th.Tensor):
    rotmat = build_rotmat(transform_xya[2])
    traslmat = build_traslmat(transform_xya[:2])
    rototrasl = th.matmul(traslmat,rotmat)
    point_num = points_Nxy.size()[0]
    # rotmat = th.as_tensor([[ca, -sa],
    #                        [sa,  ca]])
    # trasl = transform_xya[:2]
    # trasl = th.matmul(transform_xya[:2], rotmat)
    # tp =points_Nxy+transform_xya[:2]
    # ggLog.info(f"points_Nxy.size() = {points_Nxy.size()}")
    # ggLog.info(f"transform_xya[:2].size() = {transform_xya[:2].size()}")
    # ggLog.info(f"tp.size() = {tp.size()}")
    rototrasls = rototrasl.unsqueeze(0).expand(point_num,3,3)
    # ggLog.info(f"rotmats.size() = {rotmats.size()}")
    # r = th.matmul(rotmats,tp)
    # ggLog.info(f"r.size() = {r.size()}")
    # ggLog.info(f"translated_points.size() = {translated_points.size()}")
    # ggLog.info(f"rototrasl = {rototrasl}")
    points_homogeneous_Nxy1 = points_Nxy.new_ones(size=(points_Nxy.size()[0],3))
    points_homogeneous_Nxy1[:,:2] = points_Nxy
    # ggLog.info(f"rototrasl = {rototrasl.device}")
    # ggLog.info(f"points_homogeneous_Nxy1 = {points_homogeneous_Nxy1.device}")
    r = th.vmap(th.mv)(rototrasls, points_homogeneous_Nxy1)[:,:2]
    return r

def draw_shapes(images_bchw, shapes_BNxy : th.Tensor, transform_Bxya : th.Tensor, color_Brgb : th.Tensor):
    """ Draws a polygon on each image. Each polygon is represented by a list of xy points.
        """
    batch_size = shapes_BNxy.size()[0]
    poly_size = shapes_BNxy.size()[1]
    shapes_transformed_bnxy : th.Tensor = th.vmap(transform_points)(shapes_BNxy, transform_Bxya)
    shapes_transformed_bnxy[:,:,1] *= -1 # flip y axis
    shapes_transformed_bnxy *= images_bchw.size()[3]/2
    shapes_transformed_bnxy += th.as_tensor(images_bchw.size()[2:4], device=images_bchw.device)/2.0
    # ggLog.info(f"drawing {shapes_transformed_bnxy}")
    images_bchw = kornia.utils.draw_convex_polygon(images_bchw, shapes_transformed_bnxy, color_Brgb)
    return images_bchw
    # poly = th.tensor([[[4, 4], [12, 4], [12, 8], [4, 8]]])
    # color = th.tensor([[1, 1, 1]])
    # images_bchw = kornia.utils.draw_convex_polygon(images_bchw, poly, color)
    # ggLog.info(f"images_bchw.count_nonzero = {images_bchw.count_nonzero()}")
    
def transform_transform(transform1_xya, transform2_xya):
    r = th.zeros_like(transform1_xya)
    r[2] = transform1_xya[2] + transform2_xya[2] # angle is the sum of all the angles
    # traslation is the traslation1, rotated with the total rotation, plus previous translations
    r[:2] = transform_points(transform1_xya[:2].unsqueeze(0),r).squeeze() + transform2_xya[:2]
    return r

def chain_transforms(transforms_Nxya : th.Tensor):
    trans_num = transforms_Nxya.size()[0]
    transforms_chained_Nxya = th.zeros_like(transforms_Nxya)
    transforms_chained_Nxya[0] = transforms_Nxya[0]
    for i in range(1,trans_num):
        # transform the translation part with the transform up to now and sum the angle with the angle up to now
        transforms_chained_Nxya[i] = transform_transform(transforms_Nxya[i], transforms_chained_Nxya[i-1])
    return transforms_chained_Nxya
        
def draw_chain(images_bchw : th.Tensor,
               shapes_BNxy : list[th.Tensor],
               joints_BNax : th.Tensor,
               chain_colors_Brgb : list[th.Tensor],
               origin_xya : th.Tensor | None = None,
               scale : float | None = None):
    # ggLog.info(f"\n\n---------------------------------------------- Drawing chain ------------------------------------------------")

    if origin_xya is None:
        origin_xya = th.tensor([0., 0., 0.])
    if scale is None:
        scale = 0.0
    batch_size = shapes_BNxy[0].size()[0]
    chain_len = len(shapes_BNxy)
    # ggLog.info(f"joints_BNax = {joints_BNax}")
    relative_transforms_BNxya = joints_BNax.new_zeros(size=(batch_size,chain_len+1,3))
    relative_transforms_BNxya[:,0] = origin_xya
    relative_transforms_BNxya[:,1:,0] = joints_BNax[:,:,1]
    relative_transforms_BNxya[:,1:,2] = joints_BNax[:,:,0]
    # ggLog.info(f"relative_transforms_BNxya = {relative_transforms_BNxya.device}")
    abs_transforms_bnxya : th.Tensor = th.vmap(chain_transforms)(relative_transforms_BNxya)
    # ggLog.info(f"abs_transforms_bnxya = {abs_transforms_bnxya.device}")
    abs_transforms_bnxya = abs_transforms_bnxya[:,1:]
    # ggLog.info(f"abs_transforms_bnxya = {abs_transforms_bnxya.size()}")
    # ggLog.info(f"abs_transforms_bnxya[:,:,0:2] = {(abs_transforms_bnxya[:,:,0:2]).size()}")
    # ggLog.info(f"abs_transforms_bnxya[:,:,0:2]*scale = {(abs_transforms_bnxya[:,:,0:2]*scale).size()}")
    # ggLog.info(f"origin_xy = {origin_xy.size()}")
    # ggLog.info(f"shapes = {shapes_BNxy}")
    # ggLog.info(f"abs_transforms_bnxya = {abs_transforms_bnxya}")
    abs_transforms_bnxya[:,:,0:2] = abs_transforms_bnxya[:,:,0:2]*scale
    # abs_transforms_bnxya = th.vmap(transform_transform)(abs_transforms_bnxya.view(batch_size*chain_len,3),
    #                                                     origin_xya.unsqueeze(0).expand(batch_size*chain_len,3)).view(batch_size,chain_len,3)
    # ggLog.info(f"abs_transforms_bnxya*scale = {abs_transforms_bnxya}")
    shapes_BNxy = [s*scale for s in shapes_BNxy]
    # ggLog.info(f"shapes*scale = {shapes_BNxy}")

    for link in range(chain_len):
        images_bchw = draw_shapes(images_bchw,shapes_BNxy=shapes_BNxy[link],transform_Bxya=abs_transforms_bnxya[:,link],color_Brgb=chain_colors_Brgb[link])
    return images_bchw

def build_rectangle_hw(height : float | th.Tensor , width : float | th.Tensor ,
                        x : float | th.Tensor =0.0, y : float | th.Tensor =0.0):
    return build_rectangle(x-width/2, y+height/2, x+width/2, y-height/2)

def build_rectangle(left,top,right,bottom):
    return th.tensor([[left,top],[right,top],[right,bottom],[left,bottom]],dtype=th.float32)

def build_rectangle_xyahw(r_xyahw):
    return transform_points(build_rectangle_hw(r_xyahw[3],r_xyahw[4],r_xyahw[0],r_xyahw[1]),
                            th.as_tensor([0.0,0.0,r_xyahw[2]], device=r_xyahw.device))

def is_in_rectangle(x,y,rect_xyahw):
    r_x, r_y, r_a,r_height,r_width = rect_xyahw
    # compute distance from the width axis of the rectangle
    # the width axis is wa*x+wb*y+wc = 0
    wa =  th.cos(r_a)
    wb = -th.sin(r_a)
    wc = -r_x*wb - r_y*wa
    w_dist = th.abs(wa*x+wb*y+wc)/th.sqrt(wa*wa+wb*wb)
    # compute distance from the height axis of the rectangle
    ha = wb
    hb = wa
    hc = -r_x*ha - r_y*hb
    h_dist = th.abs(ha*x+hb*y+hc)/th.sqrt(ha*ha + hb*hb)
    return w_dist < r_width/2 and h_dist < r_height/2

def are_rectangles_overlapping(r1_xyahw, r2_xyahw):
    r2_corners = build_rectangle_xyahw(r2_xyahw)
    for xy in r2_corners:
        if is_in_rectangle(xy[0], xy[1], r1_xyahw):
            return True
    r1_corners = build_rectangle_xyahw(r1_xyahw)
    for xy in r1_corners:
        if is_in_rectangle(xy[0], xy[1], r2_xyahw):
            return True
    return False


if __name__=="__main__":
    image_chw = th.zeros(size=(3,128,128), dtype=th.uint8)
    ggLog.info(f"image_chw = {image_chw.size()}")
    body_size = 0.1
    thigh_width = 0.08
    thigh_length = 0.3
    shin_width = 0.07
    shin_length = 0.45
    body_shape = build_rectangle_hw(body_size, body_size)
    thigh_shape = build_rectangle_hw(thigh_width, thigh_length+thigh_width, -thigh_length/2)
    shin_shape = build_rectangle_hw(shin_width, shin_length+shin_width, -shin_length/2)
    shapes_Nxy = [body_shape, thigh_shape, shin_shape]
    rpos, kpos, hpos = 0.9, 3.14159/8, 3.14159/8
    # rpos, kpos, hpos = bstate[0][self.BASE_STATE_IDXS.HIP_POS_Z], bstate[0][self.BASE_STATE_IDXS.HIP_JOINT_POS], bstate[0][self.BASE_STATE_IDXS.KNEE_JOINT_POS]
    image_chw = draw_chain(  images_bchw = image_chw.unsqueeze(0), 
                                    shapes_BNxy=[s.unsqueeze(0) for s in shapes_Nxy],
                                    # joints_BNax=th.as_tensor([[[3.14159/2, bstate[0][self.BASE_STATE_IDXS.HIP_POS_Z]],
                                    #                            [bstate[0][self.BASE_STATE_IDXS.HIP_JOINT_POS]+1.5707,  thigh_length],
                                    #                            [-bstate[0][self.BASE_STATE_IDXS.KNEE_JOINT_POS], shin_length] ]]),
                                    joints_BNax=th.as_tensor([[[0,              rpos],
                                                                [ hpos+3.14159,  thigh_length],
                                                                [-kpos,          shin_length] ]]),
                                    chain_color_Brgb=th.as_tensor([[200,0,200]], dtype=th.uint8),
                                    scale=1,
                                    origin_xya=th.tensor([0,-0.8, 3.14159/2])).squeeze()
    # ggLog.info(f"image_chw 2 = {image_chw.size()}")
    img = image_chw.permute(1,2,0)
    # ggLog.info(f"img = {img}")
    # ggLog.info(f"nonzero = {img.count_nonzero()}")
    img = img.cpu().numpy()
    import cv2
    cv2.imwrite("./leg.png", img)



    img = th.zeros(size=(3,128,128), dtype=th.uint8)
    img += 192
    shape = th.tensor([[-0.2,0.2],
                       [ 0.2,0.2],
                       [ 0.2,-0.2],
                       [-0.2,-0.1]])
    pos = th.tensor([0.1,0.1,0.0])
    img = draw_shapes(img.unsqueeze(0),
                shapes_BNxy    = shape.unsqueeze(0),
                transform_Bxya = pos.unsqueeze(0),
                color_Brgb=th.tensor([255,0,0])).squeeze()
    pos = th.tensor([-0.5,0.1,0.0])
    img = draw_shapes(img.unsqueeze(0),
                shapes_BNxy    = shape.unsqueeze(0),
                transform_Bxya = pos.unsqueeze(0),
                color_Brgb=th.tensor([255,0,0])).squeeze()
    shape = build_rectangle_hw(0.2,0.2)
    pos = th.tensor([-0.2,0.1,3.14159/4])
    img = draw_shapes(img.unsqueeze(0),
                shapes_BNxy    = shape.unsqueeze(0),
                transform_Bxya = pos.unsqueeze(0),
                color_Brgb=th.tensor([0,0,255])).squeeze()
    shape *= 0.5
    pos = th.tensor([-0.4,0.1,3.14159/4])
    img = draw_shapes(img.unsqueeze(0),
                shapes_BNxy    = shape.unsqueeze(0),
                transform_Bxya = pos.unsqueeze(0),
                color_Brgb=th.tensor([0,255,0])).squeeze()
    pos = th.tensor([0.4,-0.3,3.14159/4])
    img = draw_shapes(img.unsqueeze(0),
                shapes_BNxy    = shape.unsqueeze(0),
                transform_Bxya = pos.unsqueeze(0),
                color_Brgb=th.tensor([255,255,0])).squeeze()
    shape = build_rectangle_hw(0.05,0.05)
    poses = [th.tensor([-0.9,  0.9, 3.14159/4]),
             th.tensor([ 0.9,  0.9, 3.14159/4]),
             th.tensor([ 0.9, -0.9, 3.14159/4]),
             th.tensor([-0.9, -0.9, 3.14159/4])]
    for pos in poses:
        ggLog.info(f"drawing at pos {pos}")
        img = draw_shapes(img.unsqueeze(0),
                    shapes_BNxy    = shape.unsqueeze(0),
                    transform_Bxya = pos.unsqueeze(0),
                    color_Brgb=th.tensor([0,0,255])).squeeze()
    img = draw_shapes(img.unsqueeze(0),
                    shapes_BNxy    = shape.unsqueeze(0),
                    transform_Bxya = th.tensor([[0.9,0,0]]),
                    color_Brgb=th.tensor([255,0,0])).squeeze()    
    img = draw_shapes(img.unsqueeze(0),
                    shapes_BNxy    = shape.unsqueeze(0),
                    transform_Bxya = th.tensor([[0.0,0.9,0]]),
                    color_Brgb=th.tensor([0,255,0])).squeeze()    
    
    import cv2
    npimg = img.permute(1,2,0).cpu().numpy()
    cv2.imwrite("./test_img.png", npimg)