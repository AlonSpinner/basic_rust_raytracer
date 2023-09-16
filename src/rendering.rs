use crate::scene::{Camera, Scene,Color, Intersectable, Light};
use image::{Rgba, RgbaImage, Pixel};
use crate::vector::V3;

const BLACK : Color = Color{r: 0.0, g: 0.0, b: 0.0};

pub fn render_depth(camera: &Camera, scene: &Scene) -> RgbaImage {
    let mut depth_buffer = vec![f64::INFINITY; camera.image_size.0 * camera.image_size.1];
    let ray_bundle = camera.get_ray_bundle();

    let mut max_depth = 0.0f64;
    let mut min_depth = f64::INFINITY;

    let camera_axis = camera.pose.r.R.get_col(2);
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let ray = ray_bundle[i][j];

            for element in &scene.elements {
                let intersection = element.geometry.intersect(&ray);
                
                match intersection {
                    Some(intersection) => {
                        let z =  intersection.time_of_flight * V3::dot(ray.direction, camera_axis);
                        if z < depth_buffer[i * camera.image_size.1 + j] {
                            depth_buffer[i * camera.image_size.1 + j] = z;
                            if z > max_depth { max_depth = z;}
                            if z < min_depth { min_depth = z;}
                        }
                    },
                    None => {}
                }
            }

        }
    }
    let mut image = RgbaImage::new(camera.image_size.0 as u32, camera.image_size.1 as u32);
    //display inverse_depth
    let min_inv_depth = 1.0 / max_depth;
    let max_inv_depth = 1.0 / min_depth;
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let depth = depth_buffer[i * camera.image_size.1 + j];
            let inv_depth = 1.0 / depth;
            let mapped_inv_depth = (inv_depth - min_inv_depth) / (max_inv_depth - min_inv_depth);
            let u8_inv_depth = (mapped_inv_depth * 255.0) as u8;
            let pixel = Rgba::from_channels(u8_inv_depth, u8_inv_depth, u8_inv_depth, 255);
            image.put_pixel(i as u32, j as u32, pixel);
        }
    }
    return image;
}

pub fn render_image(camera: &Camera, scene: &Scene) -> RgbaImage {
    let mut tof_buffer = vec![f64::INFINITY; camera.image_size.0 * camera.image_size.1];
    let mut pixel_buffer = vec![BLACK; camera.image_size.0 * camera.image_size.1];
    let ray_bundle = camera.get_ray_bundle();


    let light = &scene.lights[0];
    let mut light_intensity = 0.0;
    let mut light_color = Color::default();
    let mut light_direction = V3::default();
    match light {
        Light::Directional(d_light) => {
            light_intensity = d_light.intensity.clone();
            light_direction = d_light.direction.clone();
            light_color = d_light.color.clone();
            // Now you can use `direction` for whatever you need
        },
        Light::Point(_) => panic!("Point light not supported yet"),
    }
    
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let ray = ray_bundle[i][j];

            for element in &scene.elements {
                let intersection = element.geometry.intersect(&ray);
                
                match intersection {
                    Some(intersection) => {
                        if intersection.time_of_flight < tof_buffer[i * camera.image_size.1 + j] {
                            tof_buffer[i * camera.image_size.1 + j] = intersection.time_of_flight;

                            pixel_buffer[i * camera.image_size.1 + j] = lambret_cosine_law(intersection.normal,
                                                                    -light_direction,
                                                                    light_intensity,
                                                                    light_color,
                                                                    element.material.color,
                                                                    element.material.albedo);
                        }
                    },
                    None => {}
                }
            }

        }
    }
    let mut image = RgbaImage::new(camera.image_size.0 as u32, camera.image_size.1 as u32);
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let color = pixel_buffer[i * camera.image_size.1 + j];
            image.put_pixel(i as u32, j as u32, color.into());
        }
    }
    return image;
}

fn lambret_cosine_law(surface_normal : V3, direction_to_light :V3, light_intensity : f32, light_color : Color,
    element_color : Color, element_albedo : f32) -> Color {
    
    // assert!(surface_normal.is_unit_length());
    // assert!(direction_to_light.is_unit_length());
    
    let cos_theta = V3::dot(surface_normal, direction_to_light).max(0.0) as f32;
    let light_power = light_intensity * cos_theta;
    let light_reflected = element_albedo / std::f32::consts::PI;
    element_color * light_color * light_power * light_reflected

}