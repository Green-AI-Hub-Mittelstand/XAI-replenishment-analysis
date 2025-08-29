<a name="readme-top"></a>



<br />
<div align="center">
  <h1 align="center">XAI Replenishment Analysis</h1>
  
  <p align="center">
    <a href="https://github.com/Green-AI-Hub-Mittelstand/readme_template/issues">Report Bug</a>
    ·
    <a href="https://github.com/Green-AI-Hub-Mittelstand/readme_template/issues">Request Feature</a>
  </p>

  <br />

  <p align="center">
    <a href="https://www.green-ai-hub.de">
    <img src="images/green-ai-hub-keyvisual.svg" alt="Logo" width="80%">
  </a>
    <br />
    <h3 align="center"><strong>Green-AI Hub Mittelstand</strong></h3>
    <a href="https://www.green-ai-hub.de"><u>Homepage</u></a> 
    | 
    <a href="https://www.green-ai-hub.de/kontakt"><u>Contact</u></a>
  
   
  </p>
</div>

<br/>

## About The Project

The project with the ULT AG aims at supporting material planning and order simulation at ULT AG in Löbau using AI methods.
This project represents a dashboard that works with conventional data structures for material planning and stock management to
enable SMEs to perform their own analysis. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Table of Contents
<details>
  <summary><img src="images/table_of_contents.jpg" alt="Logo" width="2%"></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li><a href="#table-of-contents">Table of Contents</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Getting Started

Clone this repository, navigate with your terminal into this repository and execute the following steps.

### Prerequisites

To run this dashboard, you only need a single piece of software:
- **[Docker Desktop](https://www.docker.com/products/docker-desktop/)**: Must be installed on your system (Windows, macOS, or Linux) and running in the background.

This is an example of how to list things you need to use the software and how to install them.
```sh
pip install -r requirements.txt
```

### Installation

Before starting the application for the first time, the Docker image must be built. This process reads the "recipe" (`Dockerfile`), gathers all the ingredients (code, libraries), and creates the finished application package (the image).

1.  **Download the project:** Make sure you have the entire project folder on your computer.
2.  **Open a terminal:** Open a terminal (Command Prompt, PowerShell, etc.) and navigate to the `dashboard` directory of this project.
3.  **Build the image:** Run the following command. This may take a few minutes the first time.

```bash
docker build -t ult-dashboard .
```

After this step, the application is "installed" on your system and ready to be launched.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Usage

Choose the method that suits you best. **Method 1 is recommended for all users.**

### Method 1: Using Scripts

For easy execution without terminal knowledge, we have prepared scripts.

-   **For Windows:** Double-click the `start.bat` file.
-   **For macOS/Linux:** Double-click the `start.sh` file.
    *(Note: If you get an error, open a terminal once in the `dashboard` folder and make the scripts executable with `chmod +x *.sh`.)*

The script starts the application in the background. The dashboard is then accessible at the following address in your web browser:

➡️ [**http://localhost:8050**](http://localhost:8050)

### Method 2: With Docker Compose in the Terminal (For Developers) ⚙️

This method gives you more control and displays live logs of the application.

1.  **Open a terminal:** Navigate to the `dashboard` directory.
2.  **Start the application:** Run the following command.
    ```bash
    docker-compose up
    ```
    *Tip: If you want to start the application in the background (as the scripts do), use `docker-compose up -d`.*

The application is also accessible at [**http://localhost:8050**](http://localhost:8050).

### Stopping the application

To stop the application and free up resources:

-   **Simple Method:** Double-click the corresponding script (`stop.bat` or `stop.sh`).
-   **Developer Method:** If you started with `docker-compose up` in the terminal, press `Ctrl` + `C`. If you used `docker-compose up -d`, run `docker-compose down`.

---

### Data Management and Using Your Own Data 📁

This project is configured to use local data folders without needing to rebuild the Docker image. This is managed via "volumes" in the `docker-compose.yml` file.

-   **Input Data (`/data`):** All CSV or Excel files that the application needs must be placed in the local `data` folder. If you want to analyze your own data, replace the sample files in this folder.
-   **Saved Models (`/saved_models`):** All machine learning models are stored in this folder. When the application trains a new model, it is saved here and will persist even after the container is stopped.

**Note on Data Preparation:**
The application expects the data in a specific CSV format. For information on converting your own Excel files, please refer to the `Excel_to_CSV_conversion.ipynb` notebook.

For the application to function correctly, the following CSV files with these columns must be present in the `/data/sample` folder (or `/data/ult` folder for ULT).

#### `artikeldaten.csv`
Contains master data for the individual items.

-   **Nr.**: Unique identifier for the item, consistent across all data (e.g., `item x`).
-   **Basiseinheitencode** (Base Unit Code): The basic unit of measure for the item (e.g., `STK` for pieces).
-   **Herkunftsland/-region** (Country/Region of Origin): Code for the country of origin (e.g., `DE`).
-   **Ursprungsland/-region** (Source Country/Region): Code for the source country (e.g., `DE`).

#### `br_maps.csv`
This file maps the product number to a parent product series.

-   **Nr.**: The item number to be mapped.
-   **Baureihe** (Product Series): The name of the product series (e.g., `series x`).

#### `cached_inventory_data.csv`
Contains planning-relevant data for inventory management and procurement.

-   **ItemCode**: Unique identifier for the item.
-   **Sicherheitsbestand** (Safety Stock): The minimum quantity that should always be in stock.
-   **Sicherh.-Zuschl. Beschaff. Zt.** (Safety Margin Procurement Time): Safety margin for the procurement time.
-   **Maximalbestand** (Maximum Stock): The maximum stock level for this item.
-   **Beschaffungszeit** (Procurement Time): The regular time required to replenish the item.
-   **Minimale Losgröße** (Minimum Lot Size): The smallest quantity that can be ordered.
-   **Maximale Losgröße** (Maximum Lot Size): The largest quantity that can be ordered at once.

#### `posten.csv`
Contains the historical storage transaction and booking data (movement data) for all items.

-   **Buchungsdatum** (Posting Date): The date the transaction occurred (in `YYYY-MM-DD` format, e.g., `2020-06-25`).
-   **Artikelnr** (Item No.): The item number of the affected item (consistent with other CSVs).
-   **Quantity** / **Menge**: The quantity of the transaction.
-   **Postenart** (Entry Type): The type of booking (e.g., `Zugang` for a positive storage movement).
-   **Source No_**: A reference number to the source of the booking or where the item was used.

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## License

Distributed under the GPL License V3. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact

Green-AI Hub Mittelstand - info@green-ai-hub.de<br />
Dr. Nijat Mehdiyev - nijat.mehdiyev@dfki.de<br />
Andreas Emrich - andreas.emrich@dfki.de<br />

Project Link: https://github.com/Green-AI-Hub-Mittelstand/XAI-replenishment-analysis

<br />
  <a href="https://www.green-ai-hub.de/kontakt"><strong>Get in touch »</strong></a>
<br />
<br />

<p align="left">
    <a href="https://www.green-ai-hub.de">
    <img src="images/green-ai-hub-mittelstand.svg" alt="Logo" width="45%">
  </a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

