//! An example using the `Iris` dataset
use perpetual::objective::Objective;
use perpetual::{Matrix, PerpetualBooster};
use polars::prelude::*;
use std::error::Error;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn Error>> {
    
    let budget = 1.0; 

    let features_and_target = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"];

    let features_and_target_arc = features_and_target
        .iter()
        .map(|s| String::from(s.to_owned()))
        .collect::<Vec<String>>()
        .into();

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .with_columns(Some(features_and_target_arc))
        .try_into_reader_with_file_path(Some("resources/Iris.csv".into()))?
        .finish()
        .unwrap();


    let id_vars: Vec<&str> = Vec::new();
    let mdf = df.unpivot(["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"], id_vars)?;

    let data = Vec::from_iter(
        mdf.select_at_idx(1)
            .expect("Invalid column")
            .f64()?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN)),
    );

    let matrix = Matrix::new(&data, df.height(), 4);

    let species_map = HashMap::from([
        ("Iris-setosa", 0.0),
        ("Iris-versicolor", 1.0),
        ("Iris-virginica", 2.0)
    ]);

    let y = Vec::from_iter(
        df.column("Species")?
            .str()?
            .into_iter()
            .map(|v| species_map.get(v.unwrap()).copied().unwrap())
    );

    // Create booster with increased budget for better convergence
    let mut model = PerpetualBooster::default()
        .set_objective(Objective::SquaredLoss)
        .set_budget(budget);

    model.fit(&matrix, &y, None)?;

    let raw_predictions = model.predict(&matrix, true);

    let predicted_classes: Vec<f64> = raw_predictions.iter().map(|&x| {
        let d0 = (x - 0.0).abs();
        let d1 = (x - 1.0).abs();
        let d2 = (x - 2.0).abs();

        if d0 <= d1 && d0 <= d2 {
            0.0
        } else if d1 <= d0 && d1 <= d2 {
            1.0
        } else {
            2.0
        }
    }).collect();

    let correct = predicted_classes.iter()
        .zip(y.iter())
        .filter(|(pred, actual)| **pred == **actual)
        .count();
    
    let accuracy = correct as f64 / y.len() as f64;

    println!("Accuracy: {:.2}%", accuracy * 100.0);

    // Print confusion matrix
    println!("\nConfusion Matrix:");
    println!("Predicted →");
    println!("Actual ↓  0    1    2");

    for true_class in 0..3 {
        print!("{}        ", true_class);
        for pred_class in 0..3 {
            let count = predicted_classes.iter()
                .zip(y.iter())
                .filter(|(p, a)| **p == pred_class as f64 && **a == true_class as f64)
                .count();
            print!("{:<4} ", count);
        }
        println!();
    }

    Ok(())
}
