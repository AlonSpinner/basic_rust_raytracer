use crate::{scene::{Camera, Scene, Color, Intersectable, Light}, geometry::Ray};
use image::{Rgba, RgbaImage, Pixel};
use crate::vector::V3;

const SHADOW_BIAS : f64 = 1e-10;

pub fn render_depth(camera: &Camera, scene: &Scene) -> RgbaImage {
    let mut depth_buffer = vec![f64::INFINITY; camera.image_size.0 * camera.image_size.1];
    let ray_bundle = camera.get_ray_bundle();

    let mut max_depth = 0.0f64;
    let mut min_depth = f64::INFINITY;

    let camera_axis = camera.pose.r.R.get_col(2);
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let ray = ray_bundle[i][j];

            if let Some(intersection) = scene.cast(&ray) {
                let z =  intersection.time_of_flight * V3::dot(ray.direction, camera_axis);
                depth_buffer[i * camera.image_size.1 + j] = z;
                if z > max_depth { max_depth = z;}
                if z < min_depth { min_depth = z;}
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
    let mut pixel_buffer = vec![Color::black(); camera.image_size.0 * camera.image_size.1];
    let ray_bundle = camera.get_ray_bundle();
    
    for i in 0..camera.image_size.0 {
        for j in 0..camera.image_size.1 {
            let ray = ray_bundle[i][j];

            for element in &scene.elements {
                if let Some(intersection) = element.geometry.intersect(&ray) {                   
                    if intersection.time_of_flight < tof_buffer[i * camera.image_size.1 + j] {
                        tof_buffer[i * camera.image_size.1 + j] = intersection.time_of_flight;

                        let mut pixel_color = Color::black();                    
                        for light in &scene.lights {
                            let light_direction : V3;
                            let light_color : Color;
                            let light_intensity : f32;
                            match light {
                                Light::Directional(dir_light) => {
                                    light_direction = dir_light.direction;
                                    light_color = dir_light.color;
                                    light_intensity = dir_light.intensity;
                                
                                    let shadow_ray = Ray::new(intersection.point + (intersection.normal * SHADOW_BIAS),
                                    -light_direction);
                                    if let Some(_) = scene.cast(&shadow_ray) {continue;}
                                
                                }
                                Light::Point(point_light) => {
                                    light_direction = (point_light.position - intersection.point).normalize();
                                    light_color = point_light.color;

                                    let r2 = (point_light.position - intersection.point).norm2() as f32;
                                    light_intensity = point_light.intensity / (4.0 * ::std::f32::consts::PI * r2);

                                    let shadow_ray = Ray::new(intersection.point + (intersection.normal * SHADOW_BIAS),
                                    -light_direction);
                                    if let Some(shadow_intersection) = scene.cast(&shadow_ray) {
                                        if shadow_intersection.time_of_flight < r2.sqrt() as f64 {continue;}
                                        }
                                }
                            }

                            pixel_color = pixel_color + lambret_cosine_law(intersection.normal,
                                                    -light_direction,
                                                    light_intensity,
                                                    light_color,
                                                    element.material.color,
                                                    element.material.albedo);
                        }
                        pixel_buffer[i * camera.image_size.1 + j] = pixel_color.clamp();
                    }
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
    (element_color * light_color * light_power * light_reflected).clamp()
}